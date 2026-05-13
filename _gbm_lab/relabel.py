"""
Re-label the processed recordings with several lookahead windows and run the
GBM pipeline against each. Hypothesis: longer windows capture more durable
moves, raising win/loss asymmetry and lifting avg_pnl into +EV territory.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.processor import build_sft


def relabel_window(window_s: float, tp_pct: float = 0.04, sl_pct: float = 0.04):
    """Rebuild SFT with a custom window and write to a window-specific path."""
    out_path = f"data/training/sft_window{int(window_s)}.jsonl"
    print(f"\n{'='*70}")
    print(f"Building SFT with window_s={window_s}, tp_pct={tp_pct}, sl_pct={sl_pct}")
    print(f"  → {out_path}")
    print('='*70)
    result = build_sft(
        out_path=out_path,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        window_s=window_s,
        min_gap_s=10.0,
        quiet=False,
    )
    return out_path, result


if __name__ == "__main__":
    # Try 300s (5min), 600s (10min), 1200s (20min)
    for w in [300, 600, 1200]:
        relabel_window(w)
