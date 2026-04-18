#!/usr/bin/env python3
"""
Audit v2 — uses CS2Bo3Feed directly to classify file quality.
We feed every snapshot through the real feed and record:
  - final state.score_a / state.score_b (series map score)
  - did someone reach 2 (BO3) or 1 (BO1)?
  - number of rounds observed (state.round_score_*)
  - book coverage for the Match Winner tokens
A file is "good" if feed converges on a scored match AND we have
book data for both Match-Winner tokens.
"""
import gzip, json, os, sys, asyncio, logging
logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from feeds.cs2_bo3 import CS2Bo3Feed

ROOT = "Test Data"

def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else open(path)

def _resolve_mw_tokens(meta_raw: dict):
    """Returns (team1_name, team2_name, team1_token, team2_token) for Match Winner."""
    tm = meta_raw.get("token_map") or {}
    if tm:
        t1_tok = t2_tok = ""
        for tok,info in tm.items():
            if info.get("market_type") == "Match Winner":
                if info.get("outcome_index") == 0: t1_tok = tok
                elif info.get("outcome_index") == 1: t2_tok = tok
        if t1_tok and t2_tok:
            return meta_raw.get("team1",""), meta_raw.get("team2",""), t1_tok, t2_tok
    for mk in meta_raw.get("markets") or []:
        if mk.get("type") == "Match Winner":
            toks = mk.get("token_ids") or []
            if isinstance(toks, str):
                try: toks = json.loads(toks.replace("'", '"'))
                except: toks = []
            outs = mk.get("outcomes") or []
            if isinstance(outs, str):
                try: outs = json.loads(outs.replace("'", '"'))
                except: outs = []
            if len(toks) >= 2 and len(outs) >= 2:
                return outs[0], outs[1], str(toks[0]), str(toks[1])
    return "", "", "", ""

async def audit_one(path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    r = {"path": path, "size_mb": round(size/1024/1024, 1), "match_id": "",
         "team1": "", "team2": "", "mw_ok": False, "lines": 0,
         "snaps": 0, "books_a": 0, "books_b": 0,
         "final_score_a": 0, "final_score_b": 0,
         "max_round_a": 0, "max_round_b": 0,
         "game_numbers": 0, "ended": False}
    team_a_tok = team_b_tok = ""
    match_id = ""
    feed = CS2Bo3Feed()
    try:
        fh = _open(path)
    except Exception as e:
        r["error"] = str(e); return r
    game_numbers = set()
    try:
        for line in fh:
            r["lines"] += 1
            if r["lines"] > 1_500_000: break
            try: d = json.loads(line)
            except: continue
            mt = d.get("message_type","")
            rr = d.get("raw") or {}
            if mt == "_META" and not match_id:
                match_id = str(rr.get("match_id",""))
                r["match_id"] = match_id
                t1n, t2n, t1t, t2t = _resolve_mw_tokens(rr)
                r["team1"], r["team2"] = t1n, t2n
                team_a_tok, team_b_tok = t1t, t2t
                r["mw_ok"] = bool(team_a_tok and team_b_tok)
                if match_id:
                    await feed.subscribe_match(match_id)
            elif mt == "SNAPSHOT_MATCH_UPDATE":
                if not match_id:
                    match_id = str(rr.get("match_id",""))
                    if match_id: r["match_id"] = match_id; await feed.subscribe_match(match_id)
                if match_id:
                    r["snaps"] += 1
                    try:
                        feed._process_snapshot(match_id, rr)
                    except Exception:
                        pass
                    gn = rr.get("game_number")
                    if gn: game_numbers.add(gn)
            elif mt in ("_PM_book","_PM_best_bid_ask","_PRICE_SNAPSHOT","_PM_book_fresh"):
                tok = str(rr.get("tokenId") or rr.get("token_id") or rr.get("asset_id") or "")
                if tok == team_a_tok: r["books_a"] += 1
                elif tok == team_b_tok: r["books_b"] += 1
    finally:
        try: fh.close()
        except: pass
    state = feed.get_match_state(match_id) if match_id else None
    if state:
        r["final_score_a"] = state.score_a
        r["final_score_b"] = state.score_b
        ex = state.extra or {}
        r["max_round_a"] = max(r["max_round_a"], state.round_score_a or 0)
        r["max_round_b"] = max(r["max_round_b"], state.round_score_b or 0)
    r["game_numbers"] = len(game_numbers)
    # BO3 ends at 2, BO1 ends at 1. Some small matches only go to BO1.
    r["ended"] = (r["final_score_a"] >= 2 or r["final_score_b"] >= 2 or
                  (r["game_numbers"] <= 1 and (r["final_score_a"] >= 1 or r["final_score_b"] >= 1)))
    # "good" = has MW token info + saw ≥1 complete map + has book data on at least one side
    r["quality"] = "good" if (r["mw_ok"] and r["ended"] and r["snaps"] >= 300 and
                              (r["books_a"] >= 20 or r["books_b"] >= 20)) else "poor"
    return r

async def main():
    files = []
    for root,_,fns in os.walk(ROOT):
        for fn in fns:
            if fn.startswith(".") or fn.startswith("_"): continue
            if ".enrich." in fn: continue
            if not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")): continue
            p = os.path.join(root, fn)
            if os.path.getsize(p) < 200_000: continue
            files.append(p)
    files.sort()
    print(f"Auditing {len(files)} files...\n")
    results = []
    for i,p in enumerate(files, 1):
        r = await audit_one(p)
        if not r: continue
        results.append(r)
        print(f"[{i:3}/{len(files)}] {r['quality']:4} sz={r['size_mb']:6.1f}MB snaps={r['snaps']:5} "
              f"mw={int(r['mw_ok'])} books_a={r['books_a']:5} books_b={r['books_b']:5} "
              f"score={r['final_score_a']}-{r['final_score_b']} games={r['game_numbers']} "
              f"{r['team1'][:14]:14} v {r['team2'][:14]:14}  {os.path.basename(p)[:50]}", flush=True)
    # Dedupe by match_id, keep version with best score progression
    by_mid = {}
    for r in results:
        mid = r.get("match_id") or r["path"]
        key = (r["final_score_a"] + r["final_score_b"], r["snaps"], r["books_a"] + r["books_b"])
        if mid not in by_mid or key > by_mid[mid][0]:
            by_mid[mid] = (key, r)
    results = [v[1] for v in by_mid.values()]
    good = [r for r in results if r["quality"] == "good"]
    good.sort(key=lambda x: (x["snaps"], x["books_a"] + x["books_b"]), reverse=True)
    print("\n=== SUMMARY ===")
    print(f"Total unique matches: {len(results)}")
    print(f"Good: {len(good)}")
    print(f"Poor: {len(results)-len(good)}")
    with open("_audit/audit_results_v2.json","w") as fh:
        json.dump({"all": results, "good": good}, fh, indent=2)
    with open("_audit/good_files.txt","w") as fh:
        for r in good:
            fh.write(r["path"] + "\n")
    print(f"\nWrote _audit/audit_results_v2.json + _audit/good_files.txt ({len(good)} good)")

if __name__ == "__main__":
    asyncio.run(main())
