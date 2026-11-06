"""Position sizing from risk %, stop distance and broker symbol properties."""

from __future__ import annotations

import math
from decimal import ROUND_FLOOR, Decimal

from goldbot.models import SymbolSpec


def round_volume_down(volume: float, step: float) -> float:
    """Round DOWN to a multiple of ``step`` without float artefacts (0.07/0.01 -> 7, not 6.999)."""
    volume, step = float(volume), float(step)  # numpy floats -> plain floats
    if volume <= 0 or not math.isfinite(volume):
        return 0.0
    d_step = Decimal(repr(step))
    ratio = Decimal(repr(volume)) / d_step
    steps = ratio.quantize(Decimal(1), rounding=ROUND_FLOOR)
    # tolerate tiny float error just below a step boundary (e.g. 6.9999999999)
    if ratio - steps > Decimal("0.999999"):
        steps += 1
    return float(steps * d_step)


def risk_per_lot(stop_distance: float, spec: SymbolSpec) -> float:
    """Money lost by 1.0 lot if the stop is hit, including round-turn commission."""
    return stop_distance * spec.value_per_price_unit + spec.commission_per_lot


def position_size(equity: float, risk_pct: float, stop_distance: float, spec: SymbolSpec) -> float:
    """Largest volume whose stop-out loss (incl. commission) is <= ``risk_pct`` of equity.

    Rounded down to ``volume_step`` and capped at ``volume_max``.
    Returns 0.0 if the result is below ``volume_min`` (caller must reject).
    """
    if not (equity > 0 and stop_distance > 0 and risk_pct > 0):
        return 0.0
    raw = equity * risk_pct / 100.0 / risk_per_lot(stop_distance, spec)
    vol = min(round_volume_down(raw, spec.volume_step), spec.volume_max)
    vol = round_volume_down(vol, spec.volume_step)  # volume_max may not be on the step grid
    return vol if vol >= spec.volume_min else 0.0
