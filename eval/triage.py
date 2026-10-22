"""Side-by-side triage of failures from the latest baseline dev run.
Usage: python eval/triage.py [q002 q005 ...]   (default: all failures)"""
import glob, json, sys, yaml
from run_eval import connect, run_sql, question_text

path = sorted(glob.glob("eval/results/baseline_*_dev_*.json"))[-1]
res = {r["id"]: r for r in json.load(open(path))}
qs = {q["id"]: q for q in yaml.safe_load(open("eval/questions.yaml"))}
conn = connect()
flat = lambda s: " ".join((s or "").split())

def show(rows, n=5):
    for r in rows[:n]:
        print("    ", r)
    if len(rows) > n:
        print(f"     ... {len(rows)} rows total")

print("results file:", path)
for i in sys.argv[1:] or [k for k, v in res.items() if not v["ok"]]:
    r, q = res[i], qs[i]
    print("=" * 80)
    print(i, r["category"], r["reason"])
    print("Q:", question_text(q))
    if q.get("gold_sql"):
        print("GOLD:", flat(q["gold_sql"]))
        show(run_sql(conn, q["gold_sql"]))
    print("PRED:", flat(r["pred_sql"]))
    if r["pred_sql"]:
        try:
            show(run_sql(conn, r["pred_sql"]))
        except Exception as e:
            print("     ERROR:", str(e).splitlines()[0])
