"""Provider-agnostic chat wrapper. Default: Ollama. Optional: Anthropic."""
import os
import sys
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

NUM_CTX = 8192
TEMPERATURE = 0
SEED = 42


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def _config() -> tuple[str, str]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL") or ("qwen2.5-coder:7b" if provider == "ollama" else None)
    if not model:
        raise RuntimeError(f"LLM_MODEL must be set for provider '{provider}'")
    return provider, model


def chat(system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
    provider, model = _config()
    if provider == "ollama":
        return _ollama(model, system, user, max_tokens)
    if provider == "anthropic":
        return _anthropic(model, system, user, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use 'ollama' or 'anthropic')")


def _ollama(model, system, user, max_tokens) -> LLMResponse:
    # Ollama silently truncates prompts that exceed num_ctx -> garbage SQL, no error.
    est_tokens = (len(system) + len(user)) / 3
    if est_tokens > NUM_CTX - max_tokens:
        print(f"WARNING: prompt ~{int(est_tokens)} tokens may exceed num_ctx={NUM_CTX}", file=sys.stderr)

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    t0 = time.time()
    r = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": max_tokens,
            },
        },
        timeout=600,  # cold model load on a GTX 1060 is slow
    )
    if r.status_code == 404:
        raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
    r.raise_for_status()
    d = r.json()
    return LLMResponse(
        text=d["message"]["content"],
        provider="ollama",
        model=model,
        prompt_tokens=d.get("prompt_eval_count", 0),
        completion_tokens=d.get("eval_count", 0),
        latency_s=round(time.time() - t0, 2),
    )


def _anthropic(model, system, user, max_tokens) -> LLMResponse:
    import anthropic  # lazy: optional dependency

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return LLMResponse(
        text="".join(b.text for b in resp.content if b.type == "text"),
        provider="anthropic",
        model=model,
        prompt_tokens=resp.usage.input_tokens,
        completion_tokens=resp.usage.output_tokens,
        latency_s=round(time.time() - t0, 2),
    )
