from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DTYPES = {
    "BF16": "uint8_view:bfloat16",
    "F16": "uint8_view:float16",
    "F32": "uint8_view:float32",
    "F64": "uint8_view:float64",
    "I8": "uint8_view:int8",
    "I16": "uint8_view:int16",
    "I32": "uint8_view:int32",
    "I64": "uint8_view:int64",
    "U8": "uint8_view:uint8",
    "BOOL": "uint8_view:bool",
}


class LazySafetensorsRef:
    def __init__(self, path: Path, name: str, metadata: dict[str, Any], data_offset: int):
        self.path = path
        self.name = name
        self.metadata = metadata
        self.data_offset = data_offset

    def load(self, device: str = "cpu"):
        import torch

        dtype_name = self.metadata["dtype"]
        if dtype_name not in DTYPES:
            raise TypeError(f"unsupported safetensors dtype: {dtype_name}")
        start, end = self.metadata["data_offsets"]
        payload_size = end - start
        if device == "cpu":
            with self.path.open("rb") as handle:
                handle.seek(self.data_offset + start)
                payload = bytearray(payload_size)
                view = memoryview(payload)
                position = 0
                while position < payload_size:
                    read_size = handle.readinto(view[position:])
                    if not read_size:
                        raise OSError(f"truncated safetensors tensor: {self.name}")
                    position += read_size
            tensor = torch.frombuffer(payload, dtype=torch.uint8)
        else:
            tensor = torch.empty(payload_size, dtype=torch.uint8, device=device)
            staging = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
            staging_view = memoryview(staging.numpy())
            with self.path.open("rb") as handle:
                handle.seek(self.data_offset + start)
                position = 0
                while position < payload_size:
                    requested_size = min(staging.numel(), payload_size - position)
                    read_size = handle.readinto(staging_view[:requested_size])
                    if not read_size:
                        raise OSError(f"truncated safetensors tensor: {self.name}")
                    target = tensor[position : position + read_size]
                    target.copy_(staging[:read_size], non_blocking=False)
                    position += read_size
            del staging, staging_view
        _, target_dtype = DTYPES[dtype_name].split(":")
        tensor = tensor.view(getattr(torch, target_dtype))
        return tensor.reshape(self.metadata["shape"]).contiguous()


class LazySafetensorsStateDict(Mapping):
    def __init__(self, path: str):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little", signed=False)
            header = json.loads(handle.read(header_length))
        self.data_offset = 8 + header_length
        self.entries = {name: value for name, value in header.items() if name != "__metadata__"}

    def __getitem__(self, key: str) -> LazySafetensorsRef:
        return LazySafetensorsRef(self.path, key, self.entries[key], self.data_offset)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _lazy_load_state_dict(checkpoint_file, *args: Any, **kwargs: Any):
    return dict(LazySafetensorsStateDict(checkpoint_file))


def _low_memory_load_into_meta_model(function, *args: Any, **kwargs: Any):
    import inspect

    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    device_map = bound.arguments["device_map"] or {}
    if isinstance(device_map, dict):
        cuda_devices = sorted(
            {str(target) for target in device_map.values() if str(target).startswith("cuda")}
        )
    else:
        cuda_devices = [str(device_map)] if str(device_map).startswith("cuda") else []
    large_tensor_device = cuda_devices[0] if cuda_devices else "cpu"
    for name, reference in bound.arguments["state_dict"].items():
        payload_bytes = reference.metadata["data_offsets"][1] - reference.metadata["data_offsets"][0]
        tensor = reference.load(
            device=large_tensor_device if payload_bytes > 32 * 1024 * 1024 else "cpu"
        )
        bound.arguments["state_dict"] = {name: tensor}
        disk_offload_index, cpu_offload_index = function(*bound.args, **bound.kwargs)
    return disk_offload_index, cpu_offload_index


@contextmanager
def low_memory_safetensors_loading(enabled: bool = True):
    import transformers.modeling_utils as modeling_utils

    if not enabled:
        yield
        return

    original_load = modeling_utils.load_state_dict
    original_load_into = modeling_utils._load_state_dict_into_meta_model
    modeling_utils.load_state_dict = _lazy_load_state_dict
    modeling_utils._load_state_dict_into_meta_model = (
        lambda *args, **kwargs: _low_memory_load_into_meta_model(
            original_load_into,
            *args,
            **kwargs,
        )
    )
    try:
        yield
    finally:
        modeling_utils.load_state_dict = original_load
        modeling_utils._load_state_dict_into_meta_model = original_load_into
