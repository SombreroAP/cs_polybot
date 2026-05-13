"""Unit tests for the production MM strategy module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_strategy_production import MMStrategy, MMConfig


def test_basic_round_trip():
    """A bid fill then ask fill at the wider quote should yield positive PnL."""
    s = MMStrategy(token_id="t")
    d = s.on_book_update(0.40, 0.55, ts=0)
    assert d["action"] == "cancel_and_replace"
    assert abs(d["new_bid_price"] - 0.38) < 1e-6
    assert abs(d["new_ask_price"] - 0.57) < 1e-6
    s.on_fill("bid", 0.38)
    s.on_fill("ask", 0.57)
    assert abs(s.cash - 0.19) < 1e-6
    print("✓ test_basic_round_trip")


def test_narrow_spread_no_quote():
    """Spread < min_spread should produce no action when starting flat."""
    s = MMStrategy(token_id="t")
    d = s.on_book_update(0.49, 0.51, ts=0)  # 2c spread, below 10c min
    assert d is None
    print("✓ test_narrow_spread_no_quote")


def test_drift_filter_pulls_quotes():
    """Large recent mid drift should cancel quotes."""
    cfg = MMConfig(drift_mode="absolute", max_recent_drift=0.01)
    s = MMStrategy(token_id="t", cfg=cfg)
    d = s.on_book_update(0.40, 0.55, ts=0)
    assert d["action"] == "cancel_and_replace"
    # Move the mid by 2¢ within 30s — absolute drift filter (1¢) should pull
    d = s.on_book_update(0.42, 0.57, ts=10)
    assert d is not None
    assert d["action"] == "cancel_all"
    assert d["reason"] == "drift_filter"
    print("✓ test_drift_filter_pulls_quotes (absolute mode)")


def test_relative_drift_filter():
    """Relative drift = K × spread."""
    cfg = MMConfig(drift_mode="relative", drift_ratio=0.10)
    s = MMStrategy(token_id="t", cfg=cfg)
    d = s.on_book_update(0.40, 0.55, ts=0)  # spread 15c, threshold = 1.5c
    assert d["action"] == "cancel_and_replace"
    # 1¢ drift, below 1.5¢ threshold → quote stays
    d = s.on_book_update(0.41, 0.56, ts=5)
    assert d is None or d.get("action") == "cancel_and_replace"
    # 3¢ drift, above 1.5¢ → cancel
    s = MMStrategy(token_id="t", cfg=cfg)
    s.on_book_update(0.40, 0.55, ts=0)
    d = s.on_book_update(0.43, 0.58, ts=10)
    assert d is not None
    assert d["action"] == "cancel_all"
    assert d["reason"] == "drift_filter"
    print("✓ test_relative_drift_filter")


def test_price_clamping_extreme_low_bid():
    """When best_bid is near 0, the behind-inside bid would go negative; should suppress."""
    s = MMStrategy(token_id="t")
    d = s.on_book_update(0.01, 0.99, ts=0)
    assert d is not None
    # Either no bid placed or bid >= 0.01
    if d["action"] == "cancel_and_replace" and d.get("new_bid_price") is not None:
        assert d["new_bid_price"] >= 0.01
    print("✓ test_price_clamping_extreme_low_bid")


def test_price_clamping_extreme_high_ask():
    """When best_ask is near 1, the behind-inside ask would exceed 1; should suppress."""
    s = MMStrategy(token_id="t")
    d = s.on_book_update(0.01, 0.99, ts=0)
    assert d is not None
    if d["action"] == "cancel_and_replace" and d.get("new_ask_price") is not None:
        assert d["new_ask_price"] <= 0.99
    print("✓ test_price_clamping_extreme_high_ask")


def test_inventory_cap_prevents_one_sided_quote():
    """At max inventory, should not place a bid that would push us over."""
    cfg = MMConfig(max_inv=2.0, quote_size=1.0)
    s = MMStrategy(token_id="t", cfg=cfg)
    s.on_book_update(0.40, 0.55, ts=0)
    s.on_fill("bid", 0.38)  # inv = 1
    s.on_fill("bid", 0.38)  # inv = 2
    d = s.on_book_update(0.40, 0.55, ts=100)  # need fresh past 30s window
    # At inventory 2, new bid would push to 3 > max_inv 2 → no new bid
    if d and d["action"] == "cancel_and_replace":
        assert d.get("new_bid_price") is None
    print("✓ test_inventory_cap_prevents_one_sided_quote")


def test_mark_to_market():
    s = MMStrategy(token_id="t")
    s.on_book_update(0.40, 0.55, ts=0)
    s.on_fill("bid", 0.38)
    # Now inv=1, cash=-0.38; mark to market at bid 0.40 → -0.38 + 0.40 = +0.02
    mtm = s.mark_to_market(0.40, 0.55)
    assert abs(mtm - 0.02) < 1e-6
    print("✓ test_mark_to_market")


if __name__ == "__main__":
    test_basic_round_trip()
    test_narrow_spread_no_quote()
    test_drift_filter_pulls_quotes()
    test_relative_drift_filter()
    test_price_clamping_extreme_low_bid()
    test_price_clamping_extreme_high_ask()
    test_inventory_cap_prevents_one_sided_quote()
    test_mark_to_market()
    print("\nAll 8 tests passed.")
