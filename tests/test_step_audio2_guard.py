import numpy as np
import pytest


def test_guard_is_idempotent_and_transparent_on_cpu():
    torch = pytest.importorskip("torch")
    from integration.step_audio2 import install_bf16_linear_guard

    first = install_bf16_linear_guard(32)
    second = install_bf16_linear_guard(32)
    assert first is second

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(3, 16, generator=generator)
    weight = torch.randn(7, 16, generator=generator)
    bias = torch.randn(7, generator=generator)
    expected = inputs @ weight.T + bias
    output = torch.nn.functional.linear(inputs, weight, bias)
    assert torch.allclose(output, expected)
    assert np.isfinite(output.detach().numpy()).all()
