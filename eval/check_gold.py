import os, yaml, psycopg
from dotenv import load_dotenv

load_dotenv()
conn = psycopg.connect(host=os.environ["PG_HOST"], port=os.environ["PG_PORT"], dbname=os.environ["PG_DB"], user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"])
for q in yaml.safe_load(open("eval/questions.yaml")):
    if "gold_sql" not in q:
        continue
    rows = conn.execute(q["gold_sql"]).fetchall()
    print(f"{q['id']} {q['question']}\n -> {len(rows)} rows: {rows[:5]}")
