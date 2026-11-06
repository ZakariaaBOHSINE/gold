"""Signal generation (pure functions)."""

from goldbot.strategy.trend_pullback import (
    compute_features,
    generate_signal,
    reprice_to_fill,
    signal_at,
)

__all__ = ["compute_features", "generate_signal", "reprice_to_fill", "signal_at"]
