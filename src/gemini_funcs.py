# gemini_funcs.py — adapters between RAG-Anything/LightRAG and the Gemini API.
# Also handles rate limiting, retries on 503/429, and token/cost accounting.
# To switch providers (e.g. AWS Bedrock), replacing this file is enough.
import os, asyncio, base64
import numpy as np
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc

from config import (ROOT, LLM_MODEL, VISION_MODEL, EMBED_MODEL, EMBED_DIM, MAX_RETRIES, LLM_RPM, EMBED_RPM, EMBED_TPM,
                    THINKING, PRICE_INPUT, PRICE_OUTPUT)

load_dotenv(ROOT / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Request counters: how many API calls and tokens a knowledge-base build cost
stats = {"llm": 0, "vision": 0, "embed": 0, "retries": 0,
         "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}


def cost_usd() -> float:
    """Estimated generation cost (excluding embeddings) using the prices in config.py."""
    out = stats["output_tokens"] + stats["thought_tokens"]
    return (stats["input_tokens"] * PRICE_INPUT + out * PRICE_OUTPUT) / 1e6


class RateLimiter:
    """Allows at most per_minute units per minute (a unit is a request, a text or a token)."""
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.next_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self, units: int = 1):
        async with self.lock:
            now = asyncio.get_running_loop().time()
            if now < self.next_time:
                await asyncio.sleep(self.next_time - now)
                now = self.next_time
            self.next_time = now + self.interval * units


llm_limiter = RateLimiter(LLM_RPM)      # shared by text and vision calls (same model)
embed_limiter = RateLimiter(EMBED_RPM)        # texts per minute
embed_token_limiter = RateLimiter(EMBED_TPM)  # tokens per minute


def _quota_info(e) -> str:
    """Extracts which quota was exceeded (model and quota id) from the Google error."""
    try:
        details = (e.details or {}).get("error", {}).get("details", [])
        out = []
        for d in details:
            for v in d.get("violations", []):
                model = v.get("quotaDimensions", {}).get("model", "")
                out.append(f"{v.get('quotaId', '')} {model}".strip())
            if d.get("retryDelay"):
                out.append(f"retryDelay={d['retryDelay']}")
        return "; ".join(out)
    except Exception:
        return ""


async def _with_retry(call, label=""):
    """Retries a call when the model is overloaded (5xx) or a per-minute quota is hit (429)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await call()
        except (errors.ServerError, errors.ClientError) as e:
            retryable = isinstance(e, errors.ServerError) or e.code == 429
            if not retryable or attempt == MAX_RETRIES:
                raise
            stats["retries"] += 1
            # 429 = per-minute quota: wait at least a minute; 5xx = overload: shorter backoff
            delay = min(60 * attempt, 180) if e.code == 429 else min(5 * 2 ** attempt, 120)
            info = _quota_info(e) if e.code == 429 else str(e.message)[:100]
            print(f"   ⚠️ [{label}] Gemini {e.code}, attempt {attempt}/{MAX_RETRIES}, waiting {delay} s | {info}")
            await asyncio.sleep(delay)


def _image_part(b64: str, mime: str = "image/jpeg") -> types.Part:
    return types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)


async def _generate(model, parts, system_prompt=None, kind="llm"):
    async def call():
        await llm_limiter.wait()
        stats[kind] += 1
        resp = await client.aio.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                thinking_config=types.ThinkingConfig(**THINKING) if THINKING else None,
            ),
        )
        u = resp.usage_metadata
        if u:
            stats["input_tokens"] += u.prompt_token_count or 0
            stats["output_tokens"] += u.candidates_token_count or 0
            stats["thought_tokens"] += u.thoughts_token_count or 0
        return resp.text or ""
    return await _with_retry(call, label=kind)


# ---------- LLM: text -> text ----------
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    history = "".join(f"{m['role']}: {m['content']}\n" for m in history_messages or [])
    return await _generate(LLM_MODEL, [types.Part.from_text(text=history + prompt)], system_prompt)


# ---------- Vision: text + image -> text ----------
async def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
    if messages:  # OpenAI format: [{"role":..., "content": str | [{"type":"text"}, {"type":"image_url"}]}]
        parts, sys_parts = [], []
        for m in messages:
            if not m:
                continue
            content = m.get("content")
            if m.get("role") == "system":
                sys_parts.append(content if isinstance(content, str) else "")
                continue
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
                continue
            for c in content or []:
                if c.get("type") == "text":
                    parts.append(types.Part.from_text(text=c["text"]))
                elif c.get("type") == "image_url":
                    url = c["image_url"]["url"]  # data:image/png;base64,....
                    header, b64 = url.split(",", 1)
                    mime = header[5:].split(";")[0] or "image/jpeg"
                    parts.append(_image_part(b64, mime))
        return await _generate(VISION_MODEL, parts, "\n".join(sys_parts) or system_prompt, kind="vision")
    if image_data:
        parts = [types.Part.from_text(text=prompt), _image_part(image_data)]
        return await _generate(VISION_MODEL, parts, system_prompt, kind="vision")
    return await llm_func(prompt, system_prompt, history_messages, **kwargs)


# ---------- Embeddings: texts -> vectors ----------
async def _embed(texts: list[str]) -> np.ndarray:
    approx_tokens = sum(len(t) for t in texts) // 3 + 1   # rough, conservative estimate

    async def call():
        await embed_limiter.wait(len(texts))
        await embed_token_limiter.wait(approx_tokens)
        stats["embed"] += 1
        resp = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        return np.array([e.values for e in resp.embeddings])
    return await _with_retry(call, label="embed")


embedding_func = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=2048, func=_embed)
