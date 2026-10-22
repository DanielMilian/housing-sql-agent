"""Schema context for the LLM, read live from Postgres (incl. persisted dbt descriptions)."""
import os
from functools import lru_cache

import psycopg2
from dotenv import load_dotenv

load_dotenv()

SCHEMA = "analytics"
ALLOWED_TABLES = [
    "mart_district_affordability",
    "dim_district",
    "fct_cpi_annual",
    "fct_income_district",
    "fct_rent_index",
    "fct_rent_serpavi",
]


def connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5434"),
        dbname=os.getenv("PG_DB") or os.getenv("PG_DATABASE") or "housing",
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        options="-c default_transaction_read_only=on -c statement_timeout=10000",
    )


_COLS_SQL = """
SELECT c.relname,
       obj_description(c.oid, 'pg_class'),
       a.attname,
       format_type(a.atttypid, a.atttypmod),
       col_description(c.oid, a.attnum)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE n.nspname = %s AND c.relname = ANY(%s)
ORDER BY c.relname, a.attnum
"""


@lru_cache(maxsize=8)
def get_schema_context(with_docs: bool = True, n_samples: int = 2) -> str:
    """Render allowed tables as text. with_docs=False strips dbt descriptions (ablation)."""
    tables: dict[str, dict] = {}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(_COLS_SQL, (SCHEMA, ALLOWED_TABLES))
        for rel, tdesc, col, typ, cdesc in cur.fetchall():
            t = tables.setdefault(rel, {"desc": tdesc, "cols": []})
            t["cols"].append((col, typ, cdesc))

        missing = set(ALLOWED_TABLES) - set(tables)
        if missing:
            raise ValueError(f"Tables not found / not readable by agent_ro: {sorted(missing)}")

        out = []
        for rel in ALLOWED_TABLES:  # stable order
            t = tables[rel]
            head = f"Table {SCHEMA}.{rel}"
            if with_docs and t["desc"]:
                head += f" -- {t['desc']}"
            out.append(head)
            for col, typ, cdesc in t["cols"]:
                line = f"  {col} {typ}"
                if with_docs and cdesc:
                    line += f" -- {cdesc}"
                out.append(line)
            if n_samples:
                cur.execute(f"SELECT * FROM {SCHEMA}.{rel} ORDER BY 1, 2 LIMIT {int(n_samples)}")  # rel from allowlist
                names = [d[0] for d in cur.description]
                out.append("  sample rows: " + " | ".join(names))
                for row in cur.fetchall():
                    out.append("    " + " | ".join(str(v) for v in row))
            out.append("")
    return "\n".join(out).strip()


if __name__ == "__main__":
    print(get_schema_context())
