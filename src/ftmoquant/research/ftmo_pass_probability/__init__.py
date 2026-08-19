"""FTMO 2-Step risk/sizing + Monte Carlo evaluation layer.

Sits on top of the already-frozen, validated ``usdcad_sweep_bos_retest_v1``
signal (immutable: USD/CAD.OANDA, M30, B2F1_sweep_bos_retest,
swing_lookback=40, rr=2.0). Nothing in this package changes the alpha; it
only evaluates causal position-sizing policies against FTMO 2-Step rules
using DEVELOPMENT-only block-bootstrap Monte Carlo.
"""
