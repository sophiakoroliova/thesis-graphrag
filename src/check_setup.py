# check_setup.py — quick environment check before a run (~2 API calls, under one cent).
# Checks the API key, the models from config.py, the embedding dimension,
# whether the LLM spends tokens on "thinking", and whether the data is in place.
#
#   python src/check_setup.py
import os
import sys

from dotenv import load_dotenv

import config

load_dotenv(config.ROOT / ".env")


def main():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("❌ GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")
    print("✅ API key found")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    resp = client.models.generate_content(
        model=config.LLM_MODEL,
        contents="Answer in one sentence: what is a basis of a vector space?",
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(**config.THINKING) if config.THINKING else None,
        ),
    )
    u = resp.usage_metadata
    print(f"✅ LLM {config.LLM_MODEL}: \"{resp.text.strip()[:90]}...\"")
    print(f"   tokens: input={u.prompt_token_count}, output={u.candidates_token_count}, "
          f"thinking={u.thoughts_token_count or 0}")

    emb = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=["linear independence"],
        config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
    )
    dim = len(emb.embeddings[0].values)
    ok = "✅" if dim == config.EMBED_DIM else "❌"
    print(f"{ok} embeddings {config.EMBED_MODEL}: dimension {dim} (expected {config.EMBED_DIM})")

    for path, what in [(config.PARSED_DIR, "parsed corpus"), (config.RAW_PDF, "full textbook")]:
        print(f"{'✅' if path.exists() else '⚠️ '} {what}: {path.relative_to(config.ROOT)}"
              + ("" if path.exists() else " — not found"))


if __name__ == "__main__":
    main()
