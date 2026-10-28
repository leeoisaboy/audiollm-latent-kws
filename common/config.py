from __future__ import annotations

FRAME_S = 0.08
READ_FRAME_S = 0.04
LATENT_LAYER = 2
LATENT_WIDTH = 3584
PRIMARY_K = 8
DEFAULT_CALIBRATION = "center"
DEFAULT_POOL_TOP = 20
GRID_SNAP_TOL = 6
LAMBDA_LM = 4.0
Z_EPS = 1e-8

NEUTRAL_KEYWORDS = ["开灯", "关窗", "拉帘", "调温", "打开电视", "关闭风扇"]


def is_cjk(symbol: str) -> bool:
    if not symbol:
        return False
    code = ord(symbol[0])
    return 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
