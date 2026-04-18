#!/usr/bin/env python3
"""
Audit recording files for quality:
 - has _META
 - has Match Winner token map with both outcomes
 - number of SNAPSHOT_MATCH_UPDATE
 - number of orderbook messages
 - has a proper winner (someone >= 2 maps in BO3, 1 in BO1)
 - first/last book timestamp span > 300s (actually live)
"""
import gzip, json, os, sys
from collections import defaultdict

ROOT = "Test Data"

def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else open(path)

def audit(path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    r = {
        "path": path, "size_mb": round(size/1024/1024, 1),
        "lines": 0, "meta": False, "team1": "", "team2": "", "match_id": "",
        "mw_tokens": 0, "snaps": 0, "books": 0, "book_ts_span": 0,
        "max_a": 0, "max_b": 0, "ended": False,
    }
    first_book = None; last_book = None
    try:
        fh = _open(path)
    except Exception as e:
        r["error"] = str(e); return r
    try:
        for line in fh:
            r["lines"] += 1
            if r["lines"] > 2_000_000:
                break  # bound
            try:
                d = json.loads(line)
            except Exception:
                continue
            mt = d.get("message_type","")
            rr = d.get("raw") or {}
            if mt == "_META":
                r["meta"] = True
                r["team1"] = rr.get("team1","")
                r["team2"] = rr.get("team2","")
                r["match_id"] = str(rr.get("match_id",""))
                tm = rr.get("token_map") or {}
                mw = [tok for tok,info in tm.items() if info.get("market_type") == "Match Winner"]
                r["mw_tokens"] = len(mw)
            elif mt == "SNAPSHOT_MATCH_UPDATE":
                r["snaps"] += 1
                sa = rr.get("team1_score") or rr.get("score_a") or 0
                sb = rr.get("team2_score") or rr.get("score_b") or 0
                try: sa = int(sa)
                except: sa = 0
                try: sb = int(sb)
                except: sb = 0
                if sa > r["max_a"]: r["max_a"] = sa
                if sb > r["max_b"]: r["max_b"] = sb
            elif mt in ("_PM_book","_PM_best_bid_ask","_PRICE_SNAPSHOT","_PM_book_fresh"):
                r["books"] += 1
                ts = d.get("ts") or 0
                if ts:
                    first_book = ts if first_book is None else min(first_book, ts)
                    last_book = ts if last_book is None else max(last_book, ts)
    finally:
        try: fh.close()
        except: pass
    if first_book and last_book:
        r["book_ts_span"] = round(last_book - first_book, 1) if isinstance(first_book,(int,float)) else 0
    # heuristics
    r["ended"] = r["max_a"] >= 2 or r["max_b"] >= 2 or (r["max_a"] + r["max_b"] >= 1 and r["snaps"] > 100 and (r["max_a"] > r["max_b"] or r["max_b"] > r["max_a"]))
    r["quality"] = "good" if (r["meta"] and r["mw_tokens"] >= 2 and r["snaps"] >= 300 and r["books"] >= 50 and r["ended"]) else "poor"
    return r

def main():
    files = []
    for root,_,fns in os.walk(ROOT):
        for fn in fns:
            if fn.startswith(".") or fn.startswith("_"): continue
            if ".enrich." in fn: continue
            if not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")): continue
            p = os.path.join(root, fn)
            if os.path.getsize(p) < 200_000:  # <200KB = probably incomplete
                continue
            files.append(p)
    files.sort()
    print(f"Auditing {len(files)} files...")
    results = []
    for i,p in enumerate(files, 1):
        r = audit(p)
        if not r: continue
        results.append(r)
        print(f"[{i:3}/{len(files)}] {r['quality']:4} {r['size_mb']:6.1f}MB snaps={r['snaps']:5} books={r['books']:5} "
              f"score={r['max_a']}-{r['max_b']} mw_tok={r['mw_tokens']} "
              f"{r['team1']} vs {r['team2']}  [{os.path.basename(p)}]", flush=True)
    # Deduplicate by match_id — keep largest
    by_mid = {}
    for r in results:
        mid = r["match_id"] or r["path"]
        if mid not in by_mid or r["size_mb"] > by_mid[mid]["size_mb"]:
            by_mid[mid] = r
    results = list(by_mid.values())
    good = [r for r in results if r["quality"] == "good"]
    good.sort(key=lambda x: (x["snaps"], x["books"]), reverse=True)
    print("\n=== SUMMARY ===")
    print(f"Total unique matches: {len(results)}")
    print(f"Good quality: {len(good)}")
    print(f"Poor quality: {len(results) - len(good)}")
    with open("_audit/audit_results.json","w") as fh:
        json.dump({"all": results, "good": good}, fh, indent=2)
    with open("_audit/good_files.txt","w") as fh:
        for r in good:
            fh.write(r["path"] + "\n")
    print(f"\nWrote _audit/audit_results.json and _audit/good_files.txt ({len(good)} good files)")

if __name__ == "__main__":
    main()
