"""
Trade executor for Polymarket esports bets.
Handles authentication, order placement, balance tracking, and position resolution.

In dry-run mode: simulates fills using real order book data from Polymarket CLOB,
tracks a virtual USDC balance, and resolves positions when matches end.
"""
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

import config
from persistence import TradeDB

logger = logging.getLogger(__name__)

STARTING_BALANCE = float(config.__dict__.get("STARTING_BALANCE", 1000.0))


def _adaptive_exit_targets(spread: float) -> tuple[float, float]:
    """
    Scale take-profit and stop-loss with the observed spread at entry.

    Rationale: we buy at ask, exit at bid. A 20% spread means we start 20% in the red.
    A fixed 7.5% TP is mathematically unreachable unless the whole market re-prices.
    This lets wide-spread trades take small profits and keeps tight-spread trades at
    their full target.

    Returns (tp_pct, sl_pct) both as fractions.
    """
    base_tp = float(getattr(config, "TAKE_PROFIT_PCT", 0.075))
    base_sl = float(getattr(config, "STOP_LOSS_PCT", 0.125))
    if spread <= 0.05:       # tight book → full targets
        return (base_tp, base_sl)
    if spread <= 0.10:
        return (0.05, base_sl)
    if spread <= 0.15:
        return (0.04, 0.15)
    if spread <= 0.20:
        return (0.03, 0.18)
    # > 20% spread: aim just above break-even, widen stop so normal spread bleed doesn't trigger it
    return (0.02, 0.22)


@dataclass
class PendingOrder:
    """A limit bid waiting to be filled."""
    order_id: str
    token_id: str
    team: str
    game: str
    market_id: str
    bid_price: float          # our limit bid price
    model_price: float        # what our model says fair value is
    amount: float             # USDC to spend if filled
    shares: float             # amount / bid_price
    timestamp: float = field(default_factory=time.time)
    signal_confidence: float = 0.0
    latency_edge_ms: float = 0.0
    market_price_at_entry: float = 0.0
    our_price_at_entry: float = 0.0
    status: str = "pending"   # "pending", "filled", "cancelled", "expired"
    strategy: str = "edge"
    analysis: dict = field(default_factory=dict)


@dataclass
class Position:
    """An open or closed betting position."""
    market_id: str
    token_id: str
    team: str
    game: str
    direction: str  # "buy"
    amount: float  # USDC wagered
    fill_price: float
    shares: float
    timestamp: float = field(default_factory=time.time)
    order_id: str = ""
    resolved: bool = False
    won: Optional[bool] = None
    pnl: float = 0.0
    signal_confidence: float = 0.0
    latency_edge_ms: float = 0.0
    market_price_at_entry: float = 0.0  # what Polymarket showed when we entered
    our_price_at_entry: float = 0.0     # what our model said
    strategy: str = "claude"            # "claude" = expert panel, "sum_to_one" = arb
    analysis: dict = field(default_factory=dict)  # Claude panel analysis for this bet
    exit_reason: str = ""               # "take_profit", "stop_loss", "time_exit", "resolution"
    trigger_event: str = ""             # what event triggered this trade
    sell_price: float = 0.0             # price we sold at
    # Spread-adaptive exit targets. Set at entry from the observed spread so wide-spread
    # trades can exit profitably — a fixed 7.5% TP is unreachable at 20% spread.
    effective_tp_pct: float = 0.0       # 0 → fall back to config.TAKE_PROFIT_PCT
    effective_sl_pct: float = 0.0       # 0 → fall back to config.STOP_LOSS_PCT
    spread_at_entry: float = 0.0        # observed spread when we filled (fraction, e.g. 0.20 = 20%)


class TradeExecutor:
    """Executes trades on Polymarket CLOB with balance tracking."""

    # Polymarket fees (as of 2024/2025)
    RESOLUTION_FEE_RATE = 0.02   # 2% on net winnings
    TAKER_FEE_RATE = 0.00        # 0% on CLOB (Polymarket subsidises)
    SLIPPAGE_BPS = 50            # 0.5% simulated slippage beyond top-of-book

    def __init__(self, dry_run: bool = True, starting_balance: float = STARTING_BALANCE):
        self.dry_run = dry_run
        self._client = None
        self._connected = False
        self.positions: list[Position] = []
        self.pending_orders: list[PendingOrder] = []
        self._trade_count = 0
        self._pending_count = 0
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self._session = requests.Session()
        self.total_fees_paid = 0.0

        # Persistent storage — restore state from previous runs
        self.db = TradeDB()
        self._load_from_db()

    def _load_from_db(self):
        """Restore positions and balance from SQLite on startup."""
        saved = self.db.load_state()
        if saved:
            self.balance = saved["balance"]
            self.starting_balance = saved["starting_balance"]
            self.total_fees_paid = saved["total_fees_paid"]
            self._trade_count = saved["trade_count"]

        rows = self.db.load_positions()
        for r in rows:
            self.positions.append(Position(
                market_id=r["market_id"], token_id=r["token_id"],
                team=r["team"], game=r["game"], direction=r["direction"],
                amount=r["amount"], fill_price=r["fill_price"],
                shares=r["shares"], timestamp=r["timestamp"],
                order_id=r["order_id"], resolved=r["resolved"],
                won=r["won"], pnl=r["pnl"],
                signal_confidence=r["signal_confidence"],
                latency_edge_ms=r["latency_edge_ms"],
                market_price_at_entry=r["market_price_at_entry"],
                our_price_at_entry=r["our_price_at_entry"],
                strategy=r.get("strategy", "claude"),
                analysis=r.get("analysis", {}),
                exit_reason=r.get("exit_reason", ""),
                trigger_event=r.get("trigger_event", ""),
                sell_price=r.get("sell_price", 0),
            ))

        if rows:
            logger.info(f"Restored {len(rows)} positions from DB | balance=${self.balance:.2f} | "
                        f"{sum(1 for p in self.positions if not p.resolved)} open")
            self._expire_stale_positions()
        else:
            # First run — persist initial state
            self._persist()

    def _expire_stale_positions(self, max_age_hours: float = 8.0):
        """Log stale positions on startup. Resolution is handled by bot's cleanup loop
        which has access to match state for smart win/loss detection."""
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        stale = [p for p in self.positions if not p.resolved and p.timestamp < cutoff]
        if stale:
            logger.warning(f"{len(stale)} positions older than {max_age_hours}h — cleanup loop will resolve them")

    def _persist(self):
        """Save current state to DB."""
        self.db.save_state(self.balance, self.starting_balance,
                           self.total_fees_paid, self._trade_count)

    def connect(self) -> bool:
        if self.dry_run:
            logger.info(f"TradeExecutor DRY-RUN mode | Balance: ${self.balance:.2f} USDC")
            self._connected = True
            return True

        if not config.POLYMARKET_PRIVATE_KEY or config.POLYMARKET_PRIVATE_KEY.startswith("0xYOUR"):
            logger.error("POLYMARKET_PRIVATE_KEY not set — cannot trade live")
            return False

        try:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                config.CLOB_API_URL,
                key=config.POLYMARKET_PRIVATE_KEY,
                chain_id=config.CHAIN_ID,
                signature_type=config.POLYMARKET_SIGNATURE_TYPE,
                funder=config.POLYMARKET_FUNDER_ADDRESS,
            )
            try:
                self._client.set_api_creds(self._client.derive_api_key())
            except Exception:
                self._client.set_api_creds(self._client.create_api_key())
            self._connected = True
            logger.info("TradeExecutor connected to Polymarket CLOB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Polymarket: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def open_positions_count(self) -> int:
        return sum(1 for p in self.positions if not p.resolved)

    def calculate_bet_amount(self, confidence: float) -> float:
        """Scale bet size by confidence. Cap at available balance."""
        if confidence < config.MIN_CONFIDENCE:
            return 0.0

        normalized = (confidence - config.MIN_CONFIDENCE) / (1.0 - config.MIN_CONFIDENCE)
        normalized = min(1.0, max(0.0, normalized))

        amount = config.MIN_BET + (config.MAX_BET - config.MIN_BET) * (normalized ** 1.5)
        amount = round(amount, 2)

        # Don't bet more than available balance
        available = self.balance - sum(p.amount for p in self.positions if not p.resolved)
        amount = min(amount, available)

        return max(0.0, amount)

    def _fetch_real_price(self, token_id: str) -> Optional[float]:
        """Fetch the real best ask price from Polymarket CLOB for realistic fill simulation."""
        try:
            resp = self._session.get(
                f"{config.CLOB_API_URL}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            asks = data.get("asks", [])
            if asks:
                return float(asks[0]["price"])
            return None
        except Exception:
            return None

    def execute_trade(
        self,
        token_id: str,
        team: str,
        game: str,
        market_id: str,
        amount: float,
        confidence: float,
        latency_edge_ms: float,
        max_price: float = 0.95,
        market_price: float = 0.0,
        our_price: float = 0.0,
        analysis: Optional[dict] = None,
        trigger_event: str = "",
        ai_tp_pct: Optional[float] = None,
        ai_sl_pct: Optional[float] = None,
    ) -> Optional[Position]:
        """Place a bet on a team winning."""
        if not self._connected:
            logger.error("Not connected — cannot execute trade")
            return None

        if self.open_positions_count() >= config.MAX_OPEN_POSITIONS:
            logger.info(f"[SIM] SKIP {team} — MAX_OPEN_POSITIONS={config.MAX_OPEN_POSITIONS} reached")
            return None

        if amount <= 0:
            logger.info(f"[SIM] SKIP {team} — bet amount <= 0 (${amount:.2f})")
            return None

        # Never bet both sides of the same match UNLESS it's a guaranteed arb
        from team_match import teams_match
        for p in self.positions:
            if p.resolved:
                continue
            # Check all open positions — match by market ID or team name overlap
            same_match = (p.market_id == market_id) or (
                not teams_match(p.team, team) and (
                    teams_match(p.team, team.replace(" ", "")) or
                    market_id == p.market_id
                )
            )
            if same_match and not teams_match(p.team, team):
                # We have a bet on the OTHER team — only allow if it's profitable arb
                combined_cost = p.fill_price + (market_price if market_price > 0 else 0.5)
                if combined_cost < 0.97:  # guaranteed profit after 2% fee
                    logger.info(f"[SIM] ARB: {team} + {p.team} = {combined_cost:.3f} < $1 — allowing hedge")
                else:
                    logger.info(f"[SIM] SKIP {team} — already betting on {p.team} in same match (no hedging)")
                    return None

        # Check balance
        if amount > self.balance:
            logger.warning(f"Insufficient balance: need ${amount:.2f}, have ${self.balance:.2f}")
            return None

        self._trade_count += 1

        if self.dry_run:
            return self._simulate_trade(
                token_id, team, game, market_id, amount, confidence,
                latency_edge_ms, max_price, market_price, our_price,
                analysis=analysis, trigger_event=trigger_event,
                ai_tp_pct=ai_tp_pct, ai_sl_pct=ai_sl_pct,
            )

        return self._live_trade(token_id, team, game, market_id, amount, confidence, latency_edge_ms, max_price)

    def _simulate_trade(
        self, token_id: str, team: str, game: str, market_id: str,
        amount: float, confidence: float, latency_edge_ms: float,
        max_price: float, market_price: float, our_price: float,
        analysis: Optional[dict] = None, trigger_event: str = "",
        ai_tp_pct: Optional[float] = None, ai_sl_pct: Optional[float] = None,
    ) -> Position:
        """Simulate buying at the real ask price from the orderbook.
        The position monitor sells at the real bid price."""
        # Fill at last trade price (most realistic) or ask as fallback
        fill_price = 0
        if hasattr(self, '_ws') and self._ws:
            ws_data = self._ws.get_price(token_id)
            if ws_data:
                if ws_data.get("last_trade", 0) > 0.01:
                    fill_price = ws_data["last_trade"]  # what someone actually paid
                elif ws_data.get("best_ask", 0) > 0.01:
                    fill_price = ws_data["best_ask"]  # fallback to ask
        # AI-driven mode: fall back to upstream market_price if WS book unavailable
        # (happens for newly-opened markets before WS subscription completes).
        if fill_price <= 0.01:
            if market_price and 0.02 <= market_price <= 0.98:
                fill_price = market_price
                logger.info(f"[SIM] FALLBACK {team} — WS empty, using upstream price ${fill_price:.3f}")
            else:
                logger.info(f"[SIM] SKIP {team} — no WS and no valid upstream price ({market_price})")
                return None
        if fill_price >= 0.995:
            logger.info(f"[SIM] SKIP {team} — fill ${fill_price:.3f} too close to 1.00")
            return None
        # High-price asymmetric-risk guard: above 80¢ the ceiling is too close
        # for a typical 10-15% TP to be achievable, but a 10% SL is still
        # affordable for the token. Session data: 9/10 worst losers were fills
        # between 0.77-0.95. Hard-skip these — qwen autonomy doesn't beat math.
        if fill_price > 0.75:
            logger.info(f"[SIM] SKIP {team} — fill ${fill_price:.3f} above 0.75 cap (asymmetric risk)")
            return None
        # Resolved-token check: only block ASK-side dust where bid has truly crashed.
        # The latency.py mechanical check on bid<1¢/>99¢ catches the worst cases; this
        # stops the spread-trap where ask is 30¢ but bid is 1¢ (token paid 0).
        if hasattr(self, '_ws') and self._ws:
            ws_check = self._ws.get_price(token_id)
            if ws_check:
                bid = ws_check.get("best_bid", 0)
                if bid > 0 and bid < 0.02 and fill_price > 0.10:
                    logger.info(f"[SIM] SKIP {team} — token resolved to 0 (bid={bid:.3f}, fill={fill_price:.3f})")
                    return None
                if bid > 0.98 and fill_price < 0.50:
                    logger.info(f"[SIM] SKIP {team} — token resolved to 1 (bid={bid:.3f}, fill={fill_price:.3f})")
                    return None

        # Reject if fill exceeds max
        if fill_price > max_price:
            logger.info(f"[SIM] SKIP {team} — fill ${fill_price:.3f} > max ${max_price:.3f}")
            return None

        shares = amount / fill_price if fill_price > 0 else 0

        # Observe spread at entry (for exit-condition grace period)
        spread_at_entry = 0.0
        if hasattr(self, '_ws') and self._ws:
            ws_data = self._ws.get_price(token_id)
            if ws_data and ws_data.get("best_bid", 0) > 0 and ws_data.get("best_ask", 0) > 0:
                spread_at_entry = ws_data["best_ask"] - ws_data["best_bid"]
        # AI-chosen TP/SL takes priority over adaptive defaults — qwen owns exits now.
        if ai_tp_pct is not None and ai_sl_pct is not None:
            eff_tp, eff_sl = ai_tp_pct, ai_sl_pct
        else:
            eff_tp, eff_sl = _adaptive_exit_targets(spread_at_entry)
        # SL-tightening: session data shows avg SL loss = -$20 vs avg TP win = +$13.
        # Clamp SL to at most 8% — losses are dominating wins by 52%.
        eff_sl = min(eff_sl, 0.08)
        # TP-guard: don't promise a TP that mathematically can't fill before the
        # token hits 1.00. At fill=0.75, max reachable gain = (1.00-0.75)/0.75 = 33%.
        # Leave 2¢ headroom so we actually exit, not resolve.
        max_reachable = max(0.0, (0.98 - fill_price) / fill_price) if fill_price > 0 else 0.0
        eff_tp = min(eff_tp, max_reachable * 0.7) if max_reachable > 0 else eff_tp
        eff_tp = max(eff_tp, 0.03)  # never below 3%

        # Deduct from balance just like a real trade
        self.balance -= amount

        position = Position(
            market_id=market_id,
            token_id=token_id,
            team=team,
            game=game,
            direction="buy",
            amount=amount,
            fill_price=fill_price,
            shares=shares,
            order_id=f"SIM-{self._trade_count}",
            signal_confidence=confidence,
            latency_edge_ms=latency_edge_ms,
            market_price_at_entry=market_price,
            our_price_at_entry=our_price,
            effective_tp_pct=eff_tp,
            effective_sl_pct=eff_sl,
            spread_at_entry=spread_at_entry,
            analysis=analysis or {},
            trigger_event=trigger_event,
        )

        self.positions.append(position)
        self.db.save_position(position)
        self._persist()

        logger.info(
            f"[SIM] BET ${amount:.2f} on {team} ({game}) @ {fill_price:.3f} | "
            f"shares={shares:.2f} | conf={confidence:.2f} | "
            f"our={our_price:.3f} vs mkt={market_price:.3f} | "
            f"balance=${self.balance:.2f} | edge_ms={latency_edge_ms:.0f}"
        )
        return position

    def resolve_position(self, position: Position, won: bool):
        """
        Resolve a position after a match ends.
        Won: shares pay out $1 each. Lost: shares worth $0.
        """
        if position.resolved:
            return

        position.resolved = True
        position.won = won

        if won:
            payout = position.shares * 1.0  # each share pays $1
            gross_profit = payout - position.amount
            # Polymarket charges 2% fee on net winnings (not on the principal)
            fee = max(0, gross_profit) * self.RESOLUTION_FEE_RATE
            self.total_fees_paid += fee
            net_payout = payout - fee
            position.pnl = net_payout - position.amount
            self.balance += net_payout
        else:
            position.pnl = -position.amount
            fee = 0.0

        status = "WIN" if won else "LOSS"
        fee_str = f" fee=${fee:.2f}" if fee > 0 else ""
        logger.info(
            f"[{status}] {position.team} ({position.game}) | "
            f"bet=${position.amount:.2f} pnl=${position.pnl:+.2f}{fee_str} | "
            f"balance=${self.balance:.2f} | "
            f"filled@{position.fill_price:.3f} conf={position.signal_confidence:.2f}"
        )
        self.db.update_position(position)
        self._persist()

    def resolve_by_match(self, match_id: str, winning_team: str):
        """Resolve all positions for a match based on which team won."""
        for p in self.positions:
            if p.market_id == match_id and not p.resolved:
                # Check if we bet on the winning team
                won = p.team.lower() == winning_team.lower()
                self.resolve_position(p, won)

    def check_exit_conditions(self, position: Position, current_bid: float) -> Optional[str]:
        """Check if a position should be auto-exited. Returns exit reason or None."""
        if position.resolved or current_bid <= 0:
            return None

        age = time.time() - position.timestamp
        pnl_pct = (current_bid - position.fill_price) / position.fill_price

        # Per-position adaptive targets (0 → fall back to config). Old positions persisted
        # before adaptive logic existed have 0 here and use the original fixed thresholds.
        tp_target = position.effective_tp_pct or float(getattr(config, "TAKE_PROFIT_PCT", 0.075))
        sl_target = position.effective_sl_pct or float(getattr(config, "STOP_LOSS_PCT", 0.125))

        # Take profit: price moved our way (always active)
        if pnl_pct >= tp_target:
            return "take_profit"

        # Grace period scales with spread: wide books take longer to settle after a fill.
        grace = 15
        if position.spread_at_entry >= 0.15:
            grace = 30

        # Emergency crash threshold is a drop BELOW the spread baseline, not below fill.
        # With 20% spread the bid sits ~20% below fill by construction, so absolute pnl_pct
        # starts at roughly -spread/fill. A real "crash" is another 30% below that.
        spread_baseline_pnl = -(position.spread_at_entry / position.fill_price) if position.fill_price > 0 else 0
        emergency_threshold = spread_baseline_pnl - 0.30

        if age < grace:
            if pnl_pct <= emergency_threshold:
                return "stop_loss"
            return None

        # Stop loss: price moved against us (after grace period)
        if pnl_pct <= -sl_target:
            return "stop_loss"

        # Emergency stop: real crash, not just spread
        if pnl_pct <= emergency_threshold:
            return "stop_loss"

        # Time exit: stale positions drag capital. Force-close after 15 min.
        if age > 900:
            return "time_exit"

        return None

    def sell_position(self, position: Position, sell_price: float, reason: str):
        """Exit a position by selling shares at current bid price."""
        if position.resolved:
            return

        position.resolved = True
        position.exit_reason = reason
        position.sell_price = sell_price

        proceeds = position.shares * sell_price
        raw_profit = proceeds - position.amount
        # Fee: 2% on profit only (no fee if loss)
        fee = max(0, raw_profit) * self.RESOLUTION_FEE_RATE
        self.total_fees_paid += fee
        net_profit = raw_profit - fee
        position.pnl = net_profit
        position.won = net_profit > 0
        self.balance += proceeds - fee

        age = time.time() - position.timestamp
        logger.info(
            f"[EXIT:{reason.upper()}] {position.team} ({position.game}) | "
            f"sell@{sell_price:.3f} (bought@{position.fill_price:.3f}) | "
            f"pnl=${net_profit:+.2f} ({net_profit/position.amount*100:+.1f}%) | "
            f"age={age:.0f}s | balance=${self.balance:.2f}"
        )
        self.db.update_position(position)
        self._persist()

    # ─── Limit Order Methods ────────────────────────────────────────────────

    def place_limit_bid(self, token_id: str, team: str, game: str,
                        market_id: str, model_price: float, amount: float,
                        confidence: float = 0, latency_edge_ms: float = 0,
                        market_price: float = 0, our_price: float = 0,
                        analysis: dict = None) -> Optional[PendingOrder]:
        """Place a simulated limit bid at model_price minus margin."""
        # Check position limits
        open_count = len([p for p in self.positions if not p.resolved]) + len(
            [o for o in self.pending_orders if o.status == "pending"])
        if open_count >= config.MAX_OPEN_POSITIONS:
            return None
        if amount > self.balance:
            return None
        if amount <= 0 or model_price <= 0.01:
            return None

        bid_price = round(model_price * (1 - config.LIMIT_BID_MARGIN), 3)
        bid_price = max(0.01, min(bid_price, 0.95))
        shares = amount / bid_price

        self._pending_count += 1
        order = PendingOrder(
            order_id=f"LIM-{self._pending_count}",
            token_id=token_id, team=team, game=game,
            market_id=market_id,
            bid_price=bid_price, model_price=model_price,
            amount=amount, shares=shares,
            signal_confidence=confidence,
            latency_edge_ms=latency_edge_ms,
            market_price_at_entry=market_price,
            our_price_at_entry=our_price,
            analysis=analysis or {},
        )
        self.pending_orders.append(order)

        # Reserve balance
        self.balance -= amount

        logger.info(
            f"[LIMIT BID] {team} ({game}) | bid=${bid_price:.3f} "
            f"(model=${model_price:.3f}) | ${amount:.2f} | "
            f"edge={abs(our_price - market_price)*100:.1f}%"
        )
        return order

    def check_limit_fill(self, order: PendingOrder, current_best_ask: float,
                          current_best_bid: float) -> bool:
        """Check if a pending limit bid should fill based on real ask prices.

        A limit bid fills when someone SELLS to us — meaning the best ask
        drops to our bid price or below. This happens when:
        1. A seller places an ask at or below our bid (they want out fast)
        2. The market corrects downward past our bid level
        """
        if order.status != "pending":
            return False

        age = time.time() - order.timestamp

        # Timeout: cancel if too old
        if age > config.LIMIT_ORDER_TIMEOUT:
            order.status = "expired"
            self.balance += order.amount  # refund reserved balance
            logger.info(f"[LIMIT EXPIRED] {order.team} | bid=${order.bid_price:.3f} | {age:.0f}s")
            return False

        # Fill condition: the best ask has dropped to or below our bid
        # This means a real seller wants to trade at our price
        if current_best_ask > 0.03 and current_best_ask <= order.bid_price + config.LIMIT_FILL_THRESHOLD:
            return self._fill_limit_order(order)

        return False

    def _fill_limit_order(self, order: PendingOrder) -> bool:
        """Convert a pending order into a real position."""
        order.status = "filled"
        self._trade_count += 1

        position = Position(
            market_id=order.market_id, token_id=order.token_id,
            team=order.team, game=order.game, direction="buy",
            amount=order.amount, fill_price=order.bid_price,
            shares=order.shares, order_id=order.order_id,
            signal_confidence=order.signal_confidence,
            latency_edge_ms=order.latency_edge_ms,
            market_price_at_entry=order.market_price_at_entry,
            our_price_at_entry=order.our_price_at_entry,
            strategy="edge",
            analysis=order.analysis,
        )
        self.positions.append(position)
        self.db.save_position(position)
        self._persist()

        logger.info(
            f"[LIMIT FILLED] {order.team} ({order.game}) | "
            f"filled@${order.bid_price:.3f} | ${order.amount:.2f} | "
            f"{order.shares:.1f} shares | balance=${self.balance:.2f}"
        )
        return True

    def get_pending_stats(self) -> list:
        """Return pending orders for dashboard."""
        return [
            {
                "team": o.team, "game": o.game,
                "bid_price": round(o.bid_price, 3),
                "model_price": round(o.model_price, 3),
                "amount": round(o.amount, 2),
                "age": round(time.time() - o.timestamp, 0),
                "timeout": config.LIMIT_ORDER_TIMEOUT,
                "status": o.status,
            }
            for o in self.pending_orders if o.status == "pending"
        ]

    def _live_trade(
        self, token_id: str, team: str, game: str, market_id: str,
        amount: float, confidence: float, latency_edge_ms: float, max_price: float,
    ) -> Optional[Position]:
        """Execute a real trade on Polymarket."""
        try:
            from py_clob_client.order import OrderArgs
            from py_clob_client.constants import BUY

            order_args = OrderArgs(price=max_price, size=amount, side=BUY, token_id=token_id)
            signed_order = self._client.create_order(order_args)
            result = self._client.post_order(signed_order, "FOK")

            if not result or not result.get("orderID"):
                logger.warning(f"Order rejected for {team} ({game})")
                return None

            fill_price = float(result.get("averagePrice", max_price))
            shares = amount / fill_price if fill_price > 0 else 0

            position = Position(
                market_id=market_id, token_id=token_id, team=team, game=game,
                direction="buy", amount=amount, fill_price=fill_price, shares=shares,
                order_id=result["orderID"], signal_confidence=confidence,
                latency_edge_ms=latency_edge_ms,
            )
            self.positions.append(position)
            self.balance -= amount
            self.db.save_position(position)
            self._persist()
            logger.info(f"[LIVE] BET ${amount:.2f} on {team} ({game}) @ {fill_price:.3f}")
            return position
        except Exception as e:
            logger.error(f"Trade execution failed for {team} ({game}): {e}")
            return None

    def _estimate_unrealized_pnl(self, position: Position) -> float:
        """Estimate unrealized PnL using WS bid price (what we'd actually get selling)."""
        if position.resolved:
            return position.pnl
        if hasattr(self, '_ws') and self._ws:
            ws_data = self._ws.get_price(position.token_id)
            if ws_data and ws_data["best_bid"] > 0.01:
                return position.shares * ws_data["best_bid"] - position.amount
        return 0.0

    def get_stats(self) -> dict:
        # Only show stats for currently-enabled games (LoL/Valorant excluded while awaiting GRID data)
        enabled = {g.strip().lower() for g in config.ENABLED_GAMES}
        positions = [p for p in self.positions if (p.game or "").lower() in enabled]
        resolved = [p for p in positions if p.resolved]
        open_positions = [p for p in positions if not p.resolved]
        wins = sum(1 for p in resolved if p.pnl > 0)
        losses = sum(1 for p in resolved if p.pnl <= 0)
        realized_pnl = sum(p.pnl for p in resolved)
        total_wagered = sum(p.amount for p in positions)
        open_cost = sum(p.amount for p in open_positions)

        # Calculate unrealized PnL for open positions (throttled — use cached prices)
        unrealized_pnl = 0.0
        for p in open_positions:
            unrealized_pnl += self._estimate_unrealized_pnl(p)

        total_pnl = realized_pnl + unrealized_pnl

        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "available": round(self.balance, 2),
            "total_trades": len(positions),
            "open_positions": len(open_positions),
            "open_cost": round(open_cost, 2),
            "open_value": round(open_cost + unrealized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "resolved": len(resolved),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(resolved) * 100, 1) if resolved else 0.0,
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_wagered": round(total_wagered, 2),
            "total_fees": round(self.total_fees_paid, 2),
            "avg_latency_edge_ms": round(
                sum(p.latency_edge_ms for p in positions) / len(positions), 1
            ) if positions else 0.0,
            "recent_trades": [
                {
                    "team": p.team,
                    "game": p.game,
                    "amount": round(p.amount, 2),
                    "fill_price": round(p.fill_price, 3),
                    "shares": round(p.shares, 2),
                    "confidence": round(p.signal_confidence, 2),
                    "our_price": round(p.our_price_at_entry, 3),
                    "market_price": round(p.market_price_at_entry, 3),
                    "edge": round(p.our_price_at_entry - p.market_price_at_entry, 3),
                    "strategy": p.strategy,
                    "pnl": round(p.pnl, 2) if p.resolved else round(self._estimate_unrealized_pnl(p), 2),
                    "won": p.won,
                    "resolved": p.resolved,
                    "time": p.timestamp,
                    "analysis": p.analysis if hasattr(p, 'analysis') and p.analysis else None,
                    "exit_reason": getattr(p, 'exit_reason', '') or '',
                    "sell_price": round(p.sell_price if p.sell_price > 0 else (p.fill_price + p.pnl / p.shares if p.resolved and p.shares > 0 else 0), 3),
                    "trigger_event": getattr(p, 'trigger_event', '') or '',
                    "age": round(time.time() - p.timestamp, 0) if not p.resolved else 0,
                }
                for p in positions[-100:]  # show more trades in history
            ],
        }
