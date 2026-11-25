# ask_kb.py — ask questions against a knowledge base.
#
#   python src/ask_kb.py                                       sample questions, rag_storage/baseline_text
#   python src/ask_kb.py "What is a basis of a vector space?"
#   python src/ask_kb.py --kb rag_storage/ontology_text "What is a basis?"
import argparse
import asyncio
import time
from pathlib import Path

from config import STORAGE_DIR, LIGHTRAG_KWARGS, QUERY_MODE, ENABLE_RERANK

DEFAULT_QUESTIONS = [
    "What does it mean for a set of vectors to be linearly independent?",
    "How is the rank of a matrix related to the dimension of its null space?",
    "What concepts should a student understand before learning about a basis of a subspace?",
    "When is a square matrix invertible? List several equivalent conditions.",
    "How are eigenvalues of a matrix found using the characteristic equation?",
]


async def open_kb(kb: Path):
    from raganything import RAGAnything, RAGAnythingConfig
    from gemini_funcs import llm_func, vision_func, embedding_func

    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir=str(kb)),
        llm_model_func=llm_func,
        vision_model_func=vision_func,
        embedding_func=embedding_func,
        lightrag_kwargs=LIGHTRAG_KWARGS,
    )
    await rag._ensure_lightrag_initialized()  # load an existing knowledge base from disk
    return rag


async def ask(rag, question: str) -> tuple[str, float]:
    t0 = time.time()
    answer = await rag.aquery(question, mode=QUERY_MODE, vlm_enhanced=False, enable_rerank=ENABLE_RERANK)
    return answer, time.time() - t0


async def run(kb: Path, questions: list[str]):
    rag = await open_kb(kb)
    results = []
    for q in questions:
        answer, sec = await ask(rag, q)
        results.append((q, answer, sec))
        print(f"\n{'=' * 80}\nQ: {q}\n({sec:.1f} s)\n{answer}")
    await rag.finalize_storages()
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Ask questions against a knowledge base")
    p.add_argument("question", nargs="*", help="question (without it: sample questions)")
    p.add_argument("--kb", type=Path, default=STORAGE_DIR / "baseline_text")
    args = p.parse_args(argv)
    questions = [" ".join(args.question)] if args.question else DEFAULT_QUESTIONS
    return asyncio.run(run(args.kb, questions))


if __name__ == "__main__":
    main()
