"""Phase 1 baseline: single-shot text-to-SQL with full schema context."""
import json
import re
import sys

from src.llm import chat
from src.schema import get_schema_context

SYSTEM = """You are a careful analytics SQL assistant for a Barcelona housing data warehouse (PostgreSQL).
Answer the user's question with ONE read-only SELECT query using only the tables and columns listed below.

Rules:
- Always schema-qualify tables (e.g. analytics.dim_district).
- Use only columns that appear in the schema. Never invent columns or tables.
- One statement only, PostgreSQL syntax, no comments.
- If the question cannot be answered from this data (data or years or metrics not covered, forecasts, opinions, anything that is not a read-only query), do not guess. Reply with exactly one line: REFUSE: <short reason>
- Otherwise reply with the query in a single ```sql code block and nothing else.

Schema:
{schema}
"""

_SQL_BLOCK = re.compile(r"```(?:sql|postgresql|pgsql)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_REFUSE = re.compile(r"^\s*REFUSE:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_output(text: str) -> dict:
    block = _SQL_BLOCK.search(text)
    refuse = _REFUSE.search(text)
    sql = block.group(1).strip() if block else None
    return {
        "sql": sql or None,
        "refused": bool(refuse) and not sql,
        "refusal_reason": refuse.group(1).strip() if refuse and not sql else None,
        "ambiguous": bool(refuse) and bool(sql),  # both REFUSE and SQL: flag for failure triage
        "parse_error": not sql and not refuse,    # also catches unclosed ``` from truncation
    }


def predict(question: str, with_docs: bool = True) -> dict:
    system = SYSTEM.format(schema=get_schema_context(with_docs=with_docs))
    resp = chat(system, question)
    return {
        **parse_output(resp.text),
        "raw": resp.text,
        "provider": resp.provider,
        "model": resp.model,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "latency_s": resp.latency_s,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which district had the lowest rent in 2023?"
    print(json.dumps(predict(q), indent=2, ensure_ascii=False))
