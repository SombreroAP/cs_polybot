#!/usr/bin/env python3
"""
Recording processor — turns raw ``data/recordings/*.jsonl`` files into:

  1. Validated reports (are they backtest-ready?)
  2. Normalized per-match files  (``data/processed/{match_id}.jsonl``)
  3. Training data in two formats:
       - SFT:  ``data/training/sft.jsonl``   (supervised fine-tune, prompt+response pairs)
       - RL:   ``data/training/rl.jsonl``    (state/action/reward tuples)

USAGE (from repo root):

  python data/processor.py ingest                   # pull VPS recordings + import Test Data/
  python data/processor.py validate                 # scan all recordings, print a report
  python data/processor.py validate --out report.json
  python data/processor.py process <file.jsonl>     # normalize one file
  python data/processor.py process-all              # process every backtest-ready file
  python data/processor.py build-sft                # derive SFT dataset from processed files
  python data/processor.py build-rl                 # derive RL dataset
  python data/processor.py summary                  # one-screen stats + top-10 trainable

See RECORDING_SCHEMA.md for the canonical recording format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from glob import glob
from typing import Any, Iterator, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RECORDINGS_DIR = os.path.join(REPO, "data", "recordings")
PROCESSED_DIR = os.path.join(REPO, "data", "processed")
TRAINING_DIR = os.path.join(REPO, "data", "training")

# Minimum number of fields we expect in a "rich" SNAPSHOT_MATCH_UPDATE.raw.
# Thin records (bot.match_recorder pre-patch output) have ~8 fields;
# rich records (direct bo3.gg payload) have 20+.
RICH_SNAPSHOT_MIN_FIELDS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileReport:
    """Validation result for one recording file."""
    path: str
    match_id: str = ""
    team_a: str = ""
    team_b: str = ""
    game: str = ""
    bo_type: int = 0
    has_meta: bool = False
    has_tokens: bool = False
    has_rich_snapshots: bool = False
    message_counts: dict[str, int] = field(default_factory=dict)
    rich_snapshot_count: int = 0
    thin_snapshot_count: int = 0
    total_lines: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    duration_s: float = 0.0
    size_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    # Verdict
    backtest_ready: bool = False
    training_ready: bool = False

    def score(self) -> str:
        if self.errors:
            return "error"
        if self.training_ready:
            return "training"
        if self.backtest_ready:
            return "backtest"
        return "thin"


def validate_file(path: str) -> FileReport:
    r = FileReport(path=path, size_bytes=os.path.getsize(path))
    counts: Counter[str] = Counter()
    try:
        with open(path) as f:
            for ln, line in enumerate(f, start=1):
                r.total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception as e:
                    r.errors.append(f"line {ln}: JSON parse: {e}")
                    if len(r.errors) > 5:
                        r.errors.append("…(truncated)")
                        break
                    continue
                mt = d.get("message_type", "?")
                counts[mt] += 1
                ts = d.get("ts", 0)
                if ts:
                    if not r.first_ts or ts < r.first_ts:
                        r.first_ts = ts
                    if ts > r.last_ts:
                        r.last_ts = ts
                raw = d.get("raw", {}) or {}
                if mt == "_META":
                    r.has_meta = True
                    r.match_id = str(raw.get("match_id", ""))
                    r.team_a = raw.get("team1", "") or raw.get("team_a", "")
                    r.team_b = raw.get("team2", "") or raw.get("team_b", "")
                    r.game = raw.get("game", "") or _infer_game_from_path(path)
                    r.bo_type = int(raw.get("bo_type", 0) or 0)
                    # Accept either of the two META token formats:
                    # - flat keys (MatchRecorder output):   token_id_a, token_id_b
                    # - markets array (curated Test Data): markets=[{type:"Match Winner", token_ids:[...]}, ...]
                    r.has_tokens = bool(raw.get("token_id_a") and raw.get("token_id_b"))
                    if not r.has_tokens:
                        for mk in raw.get("markets") or []:
                            if mk.get("type") == "Match Winner":
                                toks = mk.get("token_ids") or []
                                if isinstance(toks, str):
                                    try:
                                        toks = json.loads(toks.replace("'", '"'))
                                    except Exception:
                                        toks = []
                                if len(toks) >= 2 and toks[0] and toks[1]:
                                    r.has_tokens = True
                                    break
                    if not r.game:
                        r.game = "cs2"  # Test Data is all CS2, safe default
                elif mt == "SNAPSHOT_MATCH_UPDATE":
                    # Rich? count unique populated fields at top level
                    populated = sum(1 for v in raw.values() if v not in (None, "", 0, [], {}))
                    if populated >= RICH_SNAPSHOT_MIN_FIELDS:
                        r.rich_snapshot_count += 1
                    else:
                        r.thin_snapshot_count += 1
    except Exception as e:
        r.errors.append(f"open failed: {e}")

    r.message_counts = dict(counts)
    r.duration_s = max(0.0, r.last_ts - r.first_ts)
    r.has_rich_snapshots = (r.rich_snapshot_count > 0 and
                            r.rich_snapshot_count >= r.thin_snapshot_count)

    # Verdicts
    r.backtest_ready = (r.has_meta and r.has_tokens and r.total_lines > 50)
    r.training_ready = (r.backtest_ready and r.has_rich_snapshots and
                        r.rich_snapshot_count >= 50)
    return r


def validate_all(pattern: str = "*.jsonl") -> list[FileReport]:
    paths = sorted(glob(os.path.join(RECORDINGS_DIR, pattern)))
    reports = []
    for p in paths:
        try:
            reports.append(validate_file(p))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            reports.append(FileReport(path=p, errors=[f"validate_file crashed: {e}"]))
    return reports


def print_validation_summary(reports: list[FileReport]) -> None:
    by_score: Counter[str] = Counter(r.score() for r in reports)
    total = len(reports)
    print(f"Scanned {total} recordings in {RECORDINGS_DIR}\n")
    print(f"  {'training-ready':<20} {by_score['training']:>5}  (rich snapshots + tokens)")
    print(f"  {'backtest-ready':<20} {by_score['backtest']:>5}  (tokens OK, but snapshots thin)")
    print(f"  {'thin':<20} {by_score['thin']:>5}  (missing tokens or too short)")
    print(f"  {'error':<20} {by_score['error']:>5}")
    total_bytes = sum(r.size_bytes for r in reports)
    print(f"\n  total size: {total_bytes/1e6:.1f} MB")
    # Show top per-game
    games: Counter[str] = Counter(r.game for r in reports if r.game)
    print(f"  games: {dict(games)}")

    # Top 10 by training-ready
    trainers = [r for r in reports if r.training_ready]
    print(f"\ntop 10 training-ready (by duration):")
    for r in sorted(trainers, key=lambda x: -x.duration_s)[:10]:
        print(f"  {os.path.basename(r.path):<60} {r.rich_snapshot_count:>5}rich  "
              f"{r.duration_s/60:.1f}min  {r.team_a} vs {r.team_b}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalization — turns raw recording → processed/{match_id}.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def iter_lines(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def normalize_file(in_path: str, out_dir: str = PROCESSED_DIR) -> Optional[str]:
    """Produce a processed/{match_id}.jsonl with a uniform event stream.

    The processed file is STRICTLY ordered by ts and drops low-signal lines
    (e.g. heartbeat skips, redundant orderbook dupes). It's what downstream
    training/analysis tools should consume instead of the raw recordings.
    """
    os.makedirs(out_dir, exist_ok=True)
    rep = validate_file(in_path)
    if not rep.backtest_ready:
        return None  # can't normalize thin files

    events: list[dict] = []
    last_book_by_tok: dict[str, dict] = {}

    for d in iter_lines(in_path):
        mt = d.get("message_type")
        raw = d.get("raw", {}) or {}
        ts = d.get("ts")
        if not ts:
            continue

        if mt == "_META":
            events.append({"ts": ts, "type": "meta", "raw": raw})
        elif mt == "SNAPSHOT_MATCH_UPDATE":
            events.append({
                "ts": ts,
                "type": "snapshot",
                "map_name": raw.get("map_name"),
                "round_phase": raw.get("round_phase"),
                "round_number": raw.get("round_number"),
                "is_bomb_planted": raw.get("is_bomb_planted"),
                "game_ended": raw.get("game_ended"),
                "team_one": raw.get("team_one"),
                "team_two": raw.get("team_two"),
            })
        elif mt and mt.startswith("GAME_EVENT"):
            events.append({
                "ts": ts,
                "type": "event",
                "event_type": raw.get("event_type") or mt,
                "team": raw.get("team"),
                "description": raw.get("description"),
                "detail": raw,
            })
        elif mt == "_PM_best_bid_ask":
            # Dedupe: skip if price unchanged from last seen for this token
            tok = raw.get("tokenId")
            prev = last_book_by_tok.get(tok, {})
            if (prev.get("bestBid") == raw.get("bestBid") and
                    prev.get("bestAsk") == raw.get("bestAsk")):
                continue
            last_book_by_tok[tok] = raw
            events.append({
                "ts": ts, "type": "book",
                "tokenId": tok,
                "bid": raw.get("bestBid"), "ask": raw.get("bestAsk"),
            })
        elif mt == "_PM_book_fresh":
            # Test Data format — full orderbook snapshot. Extract best bid/ask.
            # Bids/asks may come sorted ascending OR descending depending on source.
            # Best bid = highest bid price; best ask = lowest ask price.
            tok = raw.get("tokenId") or raw.get("token_id") or raw.get("asset_id")
            bids = raw.get("bids") or []
            asks = raw.get("asks") or []
            def _tof(v):
                try: return float(v)
                except (ValueError, TypeError): return None
            bid_prices = [_tof(b.get("price")) for b in bids if isinstance(b, dict)]
            bid_prices = [p for p in bid_prices if p is not None]
            ask_prices = [_tof(a.get("price")) for a in asks if isinstance(a, dict)]
            ask_prices = [p for p in ask_prices if p is not None]
            best_bid = max(bid_prices) if bid_prices else None
            best_ask = min(ask_prices) if ask_prices else None
            if best_bid is None and best_ask is None:
                continue
            prev = last_book_by_tok.get(tok, {})
            if prev.get("bid") == best_bid and prev.get("ask") == best_ask:
                continue
            last_book_by_tok[tok] = {"bid": best_bid, "ask": best_ask}
            events.append({
                "ts": ts, "type": "book",
                "tokenId": tok, "bid": best_bid, "ask": best_ask,
            })
        elif mt == "_PM_last_trade_price":
            events.append({
                "ts": ts, "type": "trade",
                "tokenId": raw.get("tokenId"),
                "price": raw.get("price"),
            })

    if not events:
        return None

    events.sort(key=lambda e: e["ts"])
    match_id = rep.match_id or os.path.basename(in_path).split("_")[0]
    out_path = os.path.join(out_dir, f"{match_id}.jsonl")
    with open(out_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return out_path


def process_all(pattern: str = "*.jsonl") -> dict:
    paths = sorted(glob(os.path.join(RECORDINGS_DIR, pattern)))
    ok, skipped = 0, 0
    for p in paths:
        out = normalize_file(p)
        if out:
            ok += 1
        else:
            skipped += 1
    return {"processed": ok, "skipped_thin": skipped, "total": len(paths)}


# ─────────────────────────────────────────────────────────────────────────────
# Training set construction
# ─────────────────────────────────────────────────────────────────────────────

def build_sft(out_path: str = None) -> dict:
    """Build supervised fine-tuning examples.

    Method: replay each processed file. At every GAME_EVENT (trigger), snapshot
    the state. Look ahead N seconds. If price moved ≥ TP_PCT in favour, label
    {action: buy, hindsight_pnl: +Δ}. If it moved against ≥ SL_PCT, label
    {action: skip, hindsight_pnl: what_a_buy_would_have_lost}. Otherwise emit
    no example (ambiguous).
    """
    out_path = out_path or os.path.join(TRAINING_DIR, "sft.jsonl")
    os.makedirs(TRAINING_DIR, exist_ok=True)
    processed = sorted(glob(os.path.join(PROCESSED_DIR, "*.jsonl")))
    if not processed:
        print("No processed files — run `process-all` first.")
        return {"examples": 0}

    TP_PCT = 0.08   # move > +8% within window → "buy was right"
    SL_PCT = 0.08   # move > -8% within window → "buy was wrong, skip was right"
    WINDOW_S = 90.0 # lookahead window

    examples = 0
    skipped_ambiguous = 0

    with open(out_path, "w") as out:
        for pp in processed:
            events = list(iter_lines(pp))
            # Index book events by ts for fast lookahead
            books = [e for e in events if e.get("type") == "book"]
            if not books:
                continue

            for i, e in enumerate(events):
                if e.get("type") != "event":
                    continue
                # Build a state snapshot — latest snapshot + book before this event
                state_snap = _latest_of_type(events, i, "snapshot")
                state_books = _latest_books(events, i)
                if not state_snap or not state_books:
                    continue

                # Lookahead: find biggest price move in either direction within WINDOW_S
                t0 = e["ts"]
                window_end = t0 + WINDOW_S
                start_prices = {tok: b for tok, b in state_books.items()}
                best_move = 0.0
                worst_move = 0.0
                # Walk forward through books in the window
                for b in books:
                    if b["ts"] <= t0 or b["ts"] > window_end:
                        continue
                    tok = b.get("tokenId")
                    if tok not in start_prices:
                        continue
                    sp = start_prices[tok].get("ask") or 0
                    if not sp:
                        continue
                    new_bid = b.get("bid") or 0
                    pct = (new_bid - sp) / sp if sp else 0
                    if pct > best_move:
                        best_move = pct
                    if pct < worst_move:
                        worst_move = pct

                # Label
                if best_move >= TP_PCT and best_move > -worst_move:
                    label_action = "buy"
                    hindsight_pnl = round(best_move * 100, 2)
                elif worst_move <= -SL_PCT and -worst_move > best_move:
                    label_action = "skip"
                    hindsight_pnl = round(worst_move * 100, 2)
                else:
                    skipped_ambiguous += 1
                    continue

                examples += 1
                ex = {
                    "match_id": _extract_match_id(pp),
                    "decision_ts": t0,
                    "trigger": {
                        "event_type": e.get("event_type"),
                        "team": e.get("team"),
                        "description": e.get("description"),
                    },
                    "state": {
                        "round_phase": state_snap.get("round_phase"),
                        "round_number": state_snap.get("round_number"),
                        "team_one": state_snap.get("team_one"),
                        "team_two": state_snap.get("team_two"),
                        "map_name": state_snap.get("map_name"),
                    },
                    "market_before": {
                        tok: {"bid": v.get("bid"), "ask": v.get("ask")}
                        for tok, v in state_books.items()
                    },
                    "label": {
                        "action": label_action,
                        "source": "hindsight",
                        "pct_move_window": round(best_move if label_action == "buy" else worst_move, 4),
                        "window_s": WINDOW_S,
                    },
                    "hindsight_pnl_pct": hindsight_pnl,
                }
                out.write(json.dumps(ex) + "\n")

    print(f"SFT: {examples} examples written to {out_path} "
          f"({skipped_ambiguous} skipped as ambiguous)")
    return {"examples": examples, "skipped_ambiguous": skipped_ambiguous, "out": out_path}


def build_rl(out_path: str = None) -> dict:
    """Build RL trajectories — (state, action, reward, next_state) tuples.

    For now this is a stub that mirrors build_sft but emits the RL shape. We'll
    fill in per-action policy replays (what-would-deepseek-do, what-would-haiku-do)
    once we wire a batch inference mode. Training from this requires a
    reward-shaped labeler so we defer for phase 6.
    """
    out_path = out_path or os.path.join(TRAINING_DIR, "rl.jsonl")
    os.makedirs(TRAINING_DIR, exist_ok=True)
    # Placeholder — same examples, shaped for RL tooling
    sft_path = os.path.join(TRAINING_DIR, "sft.jsonl")
    if not os.path.exists(sft_path):
        print(f"Build SFT first: python data/processor.py build-sft")
        return {"trajectories": 0}
    n = 0
    with open(out_path, "w") as out, open(sft_path) as inp:
        for line in inp:
            ex = json.loads(line)
            # Flatten into RL tuple
            traj = {
                "state": ex["state"],
                "market": ex["market_before"],
                "trigger": ex["trigger"],
                "action": ex["label"]["action"],
                "reward_pct": ex["hindsight_pnl_pct"],
                "terminal": False,
                "match_id": ex["match_id"],
                "ts": ex["decision_ts"],
            }
            out.write(json.dumps(traj) + "\n")
            n += 1
    print(f"RL: {n} trajectories written to {out_path}")
    return {"trajectories": n, "out": out_path}


# ─── helpers ──────────────────────────────────────────────────────────────

def _latest_of_type(events: list[dict], upto: int, typ: str) -> Optional[dict]:
    for i in range(upto - 1, -1, -1):
        if events[i].get("type") == typ:
            return events[i]
    return None


def _latest_books(events: list[dict], upto: int) -> dict[str, dict]:
    books: dict[str, dict] = {}
    for i in range(upto - 1, -1, -1):
        e = events[i]
        if e.get("type") == "book":
            tok = e.get("tokenId")
            if tok not in books:
                books[tok] = e
            if len(books) >= 2:
                break
    return books


def _extract_match_id(processed_path: str) -> str:
    return os.path.basename(processed_path).replace(".jsonl", "")


def _infer_game_from_path(p: str) -> str:
    """Best-effort game inference from filename when _META.game is missing
    (e.g. old curated Test Data corpus which is CS2-only)."""
    low = p.lower()
    if "cs2" in low or "counter-strike" in low or "counterstrike" in low:
        return "cs2"
    if "dota" in low:
        return "dota2"
    if "lol" in low or "league" in low:
        return "lol"
    if "valorant" in low or "vlr" in low:
        return "valorant"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Ingest — pull remote recordings + import historical corpora into one place
# ─────────────────────────────────────────────────────────────────────────────

# Source of truth for where recordings live.
# Priority (first match wins when multiple sources have the same match_id):
#   1. data/recordings/       (local — whatever's already here, incl. Mac history)
#   2. rsync from VPS         (data/recordings_vps/ → merged in)
#   3. Test Data/old_recordings/  (curated corpus, already rich)
#
# After ingest every usable recording lives under data/recordings/ with a
# canonical filename. The processor then treats them all uniformly.

TEST_DATA_DIR = os.path.join(REPO, "Test Data", "old_recordings")


def ingest(vps_host: str = "bot@85.137.174.57",
           vps_dir: str = "/home/bot/esports/data/recordings",
           import_test_data: bool = True) -> dict:
    """Pull remote VPS recordings + import historical Test Data into one corpus.

    Idempotent: safe to rerun. rsync only copies new/changed files; Test Data
    import uses hard-link when on the same filesystem (zero-copy).
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    results = {"vps_pulled": 0, "testdata_linked": 0, "errors": []}

    # 1. Rsync from VPS
    print(f"[ingest] pulling from VPS {vps_host}:{vps_dir}/ → data/recordings/")
    import subprocess
    try:
        before = set(os.listdir(RECORDINGS_DIR))
        r = subprocess.run(
            ["rsync", "-az", "--info=stats1",
             f"{vps_host}:{vps_dir}/", f"{RECORDINGS_DIR}/"],
            check=True, capture_output=True, text=True,
        )
        after = set(os.listdir(RECORDINGS_DIR))
        results["vps_pulled"] = len(after - before)
        # rsync summarizes transferred file count in its stats block
        for ln in (r.stdout or "").splitlines():
            if "Number of regular files transferred" in ln:
                print(f"[ingest]   {ln.strip()}")
    except subprocess.CalledProcessError as e:
        results["errors"].append(f"rsync failed (exit {e.returncode}): {e.stderr[:200]}")
    except FileNotFoundError:
        results["errors"].append("rsync not installed")

    # 2. Import Test Data/ (curated corpus — already rich)
    if import_test_data and os.path.isdir(TEST_DATA_DIR):
        print(f"[ingest] importing {TEST_DATA_DIR} → data/recordings/")
        for fn in sorted(os.listdir(TEST_DATA_DIR)):
            if not fn.endswith(".jsonl"):
                continue
            src = os.path.join(TEST_DATA_DIR, fn)
            # Canonicalize filename — prefix with "tt_" to mark as Test Data origin
            # (so we never accidentally overwrite live recordings with an old file)
            dst_name = "tt_" + fn if not fn.startswith("tt_") else fn
            dst = os.path.join(RECORDINGS_DIR, dst_name)
            if os.path.exists(dst):
                continue
            try:
                # Prefer hard link (zero-copy, same FS); fallback to copy
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
            results["testdata_linked"] += 1

    print(f"[ingest] done — pulled {results['vps_pulled']} from VPS, "
          f"linked {results['testdata_linked']} from Test Data/")
    if results["errors"]:
        print(f"[ingest] errors:")
        for e in results["errors"]:
            print(f"   {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary — one-screen stats useful as a "do we have enough data yet?" readout
# ─────────────────────────────────────────────────────────────────────────────

def summary(pattern: str = "*.jsonl") -> None:
    reports = validate_all(pattern=pattern)
    if not reports:
        print(f"no files matched {pattern} in {RECORDINGS_DIR}")
        return
    print_validation_summary(reports)

    # Minutes of rich data per game
    mins_by_game: dict[str, float] = {}
    for r in reports:
        if r.training_ready:
            mins_by_game.setdefault(r.game or "unknown", 0.0)
            mins_by_game[r.game or "unknown"] += r.duration_s / 60.0
    if mins_by_game:
        print("\ntraining-ready duration per game:")
        for g, m in sorted(mins_by_game.items(), key=lambda kv: -kv[1]):
            print(f"  {g:<10} {m:.0f} min ({m/60:.1f} hours)")

    # Tournament diversity (from filenames — rough proxy)
    names = set()
    for r in reports:
        if r.training_ready:
            names.add(os.path.basename(r.path).split("_")[0])
    print(f"\nunique match IDs in training-ready set: {len(names)}")

    # Yardstick for "enough data to train a small adapter"
    rich_mins = sum(mins_by_game.values())
    if rich_mins < 60:
        print(f"\n⚠️  only {rich_mins:.0f} min of rich data — keep collecting")
    elif rich_mins < 600:
        print(f"\n✅ {rich_mins:.0f} min of rich data — enough for backtest regression tests")
    else:
        print(f"\n✅ {rich_mins:.0f} min of rich data — enough to start prompt tuning / small fine-tune")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="pull VPS recordings + import Test Data")
    i.add_argument("--vps", default="bot@85.137.174.57")
    i.add_argument("--vps-dir", default="/home/bot/esports/data/recordings")
    i.add_argument("--no-test-data", action="store_true",
                   help="skip linking Test Data/old_recordings")

    sub.add_parser("summary", help="one-screen stats + top training-ready files")

    v = sub.add_parser("validate", help="scan recordings, print ready/not-ready")
    v.add_argument("--out", default="", help="write JSON report to this path")
    v.add_argument("--pattern", default="*.jsonl")

    p = sub.add_parser("process", help="normalize ONE recording → processed/")
    p.add_argument("file", help="path to a data/recordings/*.jsonl")

    sub.add_parser("process-all", help="normalize every backtest-ready recording")

    sub.add_parser("build-sft", help="derive SFT dataset from processed files")
    sub.add_parser("build-rl", help="derive RL dataset from SFT")

    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest(vps_host=args.vps, vps_dir=args.vps_dir,
               import_test_data=not args.no_test_data)

    elif args.cmd == "summary":
        summary()

    elif args.cmd == "validate":
        reports = validate_all(pattern=args.pattern)
        print_validation_summary(reports)
        if args.out:
            with open(args.out, "w") as f:
                json.dump([asdict(r) for r in reports], f, indent=2)
            print(f"\nwrote full report to {args.out}")

    elif args.cmd == "process":
        out = normalize_file(args.file)
        print(f"→ {out}" if out else "skipped (not backtest-ready)")

    elif args.cmd == "process-all":
        r = process_all()
        print(f"processed {r['processed']} / skipped {r['skipped_thin']} / total {r['total']}")

    elif args.cmd == "build-sft":
        build_sft()

    elif args.cmd == "build-rl":
        build_rl()


if __name__ == "__main__":
    main()
