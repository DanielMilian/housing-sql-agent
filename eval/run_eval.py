import argparse, json, os, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import psycopg, yaml
from dotenv import load_dotenv
from compare import results_match

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `src` imports work

load_dotenv()

def connect():
    return psycopg.connect(
        host=os.environ["PG_HOST"], port=os.environ["PG_PORT"],
        dbname=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"], autocommit=True,
        options="-c statement_timeout=10000")  # LLM-generated SQL: never let it hang the run

def run_sql(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

def question_text(q):
    for k in ("question", "text", "nl", "prompt"):
        if q.get(k):
            return q[k]
    return None

def predict_gold(q):
    if q.get("expect") == "refuse":
        return {"sql": None, "refused": True}
    return {"sql": q["gold_sql"], "refused": False}

def predict_refuse_all(q):
    return {"sql": None, "refused": True}

def predict_baseline(q):
    from src.baseline import predict  # lazy: gold/refuse_all don't need Ollama or the schema
    text = question_text(q)
    if text is None:
        raise KeyError(f"{q['id']}: no question-text key found (keys: {list(q)})")
    return predict(text)

PREDICTORS = {"gold": predict_gold, "refuse_all": predict_refuse_all, "baseline": predict_baseline}
LLM_PREDICTORS = {"baseline"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="gold", choices=PREDICTORS)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--final", action="store_true",
                    help="required to run an LLM predictor on holdout/all (final report only)")
    args = ap.parse_args()
    if args.predictor in LLM_PREDICTORS and args.split != "dev" and not args.final:
        sys.exit("Refusing: LLM predictors run on dev only. Use --final for the final report.")

    questions = yaml.safe_load(open("eval/questions.yaml"))
    conn, results = connect(), []
    for q in questions:
        if args.split != "all" and q["split"] != args.split:
            continue
        ok, err, reason = False, None, None
        t0 = time.time()
        pred = PREDICTORS[args.predictor](q)
        if q.get("expect") == "refuse":
            ok = bool(pred["refused"]) and pred["sql"] is None
            if not ok:
                reason = "parse_error" if pred.get("parse_error") else "answered_unanswerable"
        else:
            gold = run_sql(conn, q["gold_sql"])
            assert gold, f"{q['id']}: gold returned no rows. Fix the question"
            if pred["refused"] or not pred["sql"]:
                err = "refused or no SQL"
                reason = "parse_error" if pred.get("parse_error") else "over_refusal"
            else:
                try:
                    ok = results_match(gold, run_sql(conn, pred["sql"]), q.get("ordered", False))
                    if not ok:
                        reason = "wrong_result"
                except Exception as e:
                    err, reason = str(e)[:200], "sql_error"
        results.append({"id": q["id"], "category": q["category"],
                        "difficulty": q.get("difficulty", "refuse"),
                        "question": question_text(q),
                        "ok": ok, "reason": reason, "error": err,
                        "seconds": round(time.time() - t0, 2),
                        "model": pred.get("model"), "pred_sql": pred.get("sql"),
                        "refused": pred.get("refused"), "refusal_reason": pred.get("refusal_reason"),
                        "ambiguous": pred.get("ambiguous"), "parse_error": pred.get("parse_error"),
                        "prompt_tokens": pred.get("prompt_tokens"),
                        "completion_tokens": pred.get("completion_tokens"),
                        "latency_s": pred.get("latency_s"), "raw": pred.get("raw")})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        for key in (r["difficulty"], r["category"], "ALL"):
            by[key][1] += 1
            by[key][0] += r["ok"]
    print(f"\npredictor={args.predictor} split={args.split}")
    for k, (p, n) in sorted(by.items()):
        print(f"  {k:<18} {p}/{n}")
    if args.predictor in LLM_PREDICTORS and results:
        print(f"  model={results[0]['model']} "
              f"prompt_tokens={sum(r['prompt_tokens'] or 0 for r in results)} "
              f"completion_tokens={sum(r['completion_tokens'] or 0 for r in results)}")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['id']} ({r['category']}) [{r['reason']}]: {r['error']}")
            if r["pred_sql"]:
                print(f"       sql: {' '.join(r['pred_sql'].split())[:300]}")

    os.makedirs("eval/results", exist_ok=True)
    tag = f"{args.predictor}" + (f"_{results[0]['model'].replace(':', '-')}" if args.predictor in LLM_PREDICTORS and results else "")
    path = f"eval/results/{tag}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(results, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"  saved {path}")

if __name__ == "__main__":
    main()
