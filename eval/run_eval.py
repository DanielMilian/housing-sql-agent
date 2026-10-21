import argparse, json, os, time
from collections import defaultdict
from datetime import datetime
import psycopg, yaml
from dotenv import load_dotenv
from compare import results_match

load_dotenv()

def connect():
    return psycopg.connect(
        host=os.environ["PG_HOST"], port=os.environ["PG_PORT"],
        dbname=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"], autocommit=True)

def run_sql(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

def predict_gold(q):
    if q.get("expect") == "refuse":
        return {"sql": None, "refused": True}
    return {"sql": q["gold_sql"], "refused": False}

def predict_refuse_all(q):
    return {"sql": None, "refused": True}

PREDICTORS = {"gold": predict_gold, "refuse_all": predict_refuse_all}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="gold", choices=PREDICTORS)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    args = ap.parse_args()

    questions = yaml.safe_load(open("eval/questions.yaml"))
    conn, results = connect(), []
    for q in questions:
        if args.split != "all" and q["split"] != args.split:
            continue
        ok, err = False, None
        t0 = time.time()
        pred = PREDICTORS[args.predictor](q)
        if q.get("expect") == "refuse":
            ok = pred["refused"] and pred["sql"] is None
        else:
            gold = run_sql(conn, q["gold_sql"])
            assert gold, f"{q['id']}: gold returned no rows. Fix the question"
            if pred["refused"] or not pred["sql"]:
                err = "refused or no SQL"
            else:
                try:
                    ok = results_match(gold, run_sql(conn, pred["sql"]), q.get("ordered", False))
                except Exception as e:
                    err = str(e)[:200]
        results.append({"id": q["id"], "category": q["category"],
                        "difficulty": q.get("difficulty", "refuse"),
                        "ok": ok, "error": err, "seconds": round(time.time() - t0, 2)})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        for key in (r["difficulty"], r["category"], "ALL"):
            by[key][1] += 1
            by[key][0] += r["ok"]
    print(f"\npredictor={args.predictor} split={args.split}")
    for k, (p, n) in sorted(by.items()):
        print(f"  {k:<18} {p}/{n}")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['id']} ({r['category']}): {r['error']}")

    os.makedirs("eval/results", exist_ok=True)
    path = f"eval/results/{args.predictor}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(results, open(path, "w"), indent=2)

if __name__ == "__main__":
    main()
