"""Evaluate the final closure rule (rule D) on the second, blind spot check (blink_spotcheck_round2.py).

    python blink_spotcheck_round2_eval.py            # needs blink_r2_spotcheck_verdicts.csv (downloaded from the sheet)

Reads results/blink_eyelid_distance/blink_r2_spotcheck_all_videos.csv (items with their stratum) and
blink_r2_spotcheck_verdicts.csv (item, human_verdict), writes the verdicts back into the item table and
blink_r2_spotcheck_result.csv: per stratum, how many items were judged fully closed / partly closed / open / cannot tell,
and for rule D on the judged items: precision (closures among the items the rule calls closed), the share of closures among
the "near" strata (what the rule misses just above its threshold) and among the "open" stratum (closures with no signal
at all). Closure = fully or partly closed; "cannot tell" items are excluded from the rates. Nothing else is modified.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402

OUT = HERE / "results" / "blink_eyelid_distance"
Q = pd.read_csv(OUT / "blink_r2_spotcheck_all_videos.csv")
V = pd.read_csv(OUT / "blink_r2_spotcheck_verdicts.csv")
Q["human_verdict"] = Q.item.map(dict(zip(V.item, V.human_verdict))).fillna("")
Q.to_csv(OUT / "blink_r2_spotcheck_all_videos.csv", index=False)
J = Q[Q.human_verdict != ""].copy()
J["closure"] = J.human_verdict.str.startswith(("eye closed", "partly"))
J["full"] = J.human_verdict.str.startswith("eye closed")
J["usable"] = ~J.human_verdict.str.startswith("cannot")

rows = []
order = ["closed", "near 0.70-0.85", "near 0.85-1.00", "open"]
for s in order + ["all"]:
    g = J if s == "all" else J[J.stratum == s]
    u = g[g.usable]
    rows.append({"stratum": s, "rule_D_says": "closed" if s == "closed" else ("open" if s != "all" else ""), "items_judged": len(g),
                 "fully_closed": int(g.full.sum()), "partly_closed": int((g.closure & ~g.full).sum()), "eye_open": int((u.closure == False).sum()),
                 "cannot_tell": int((~g.usable).sum()), "closure_share_pct": round(100 * u.closure.mean(), 1) if len(u) else float("nan")})
R = pd.DataFrame(rows)
R.to_csv(OUT / "blink_r2_spotcheck_result.csv", index=False)
pd.set_option("display.width", 200)
print(R.to_string(index=False))
u = J[J.usable]
if len(u):
    flagged = u[u.rule_D_closed]
    print(f"\nrule D on the judged items: precision {100 * flagged.closure.mean():.1f}% ({int(flagged.closure.sum())} of {len(flagged)} flagged items are closures, "
          f"{int(flagged.full.sum())} full)")
    for s in order[1:]:
        g = u[u.stratum == s]
        if len(g):
            print(f"  missed in '{s}': {int(g.closure.sum())} of {len(g)} are closures ({int(g.full.sum())} full)")
print(f"\n{len(J)} of {len(Q)} items judged; {int((~J.usable).sum())} 'cannot tell'")
