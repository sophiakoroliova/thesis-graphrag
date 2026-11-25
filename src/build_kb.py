# build_kb.py — builds the knowledge base (graph + vectors) from the parsed chapters.
#
#   python src/build_kb.py --text-only                    dry run: statistics only, no API calls
#   python src/build_kb.py --text-only --run              build from all chapters
#   python src/build_kb.py --text-only --run --chapters ch3
#   python src/build_kb.py --run                          full multimodal RAG-Anything mode
#
# The mode must be the SAME for the baseline and the ontology-guided method.
# If a build is interrupted, re-running continues from the first unfinished chapter
# (finished chapters are recorded in done_chapters.json, LLM responses in the LightRAG cache).
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from config import CHAPTERS, PARSED_DIR, STORAGE_DIR, LIGHTRAG_KWARGS
from preprocess import load_chapter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a RAG-Anything knowledge base")
    p.add_argument("--run", action="store_true", help="actually build (without it: statistics only)")
    p.add_argument("--text-only", action="store_true",
                   help="cost-saving mode: equations as LaTeX in the text, figures as captions")
    p.add_argument("--chapters", nargs="+", choices=CHAPTERS, default=CHAPTERS, help="chapters to include")
    p.add_argument("--name", default="baseline", help="configuration name (baseline, ontology, ...)")
    p.add_argument("--working-dir", type=Path, default=None,
                   help="knowledge base folder (default: rag_storage/<name>[_text])")
    p.add_argument("--parsed-dir", type=Path, default=PARSED_DIR, help="MinerU output folder")
    args = p.parse_args(argv)
    if args.working_dir is None:
        args.working_dir = STORAGE_DIR / (f"{args.name}_text" if args.text_only else args.name)
    return args


def load_done(path: Path) -> set:
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


async def build(args):
    done_file = args.working_dir / "done_chapters.json"
    done = load_done(done_file)

    total_chars = 0
    for ch in args.chapters:
        items, counts = load_chapter(args.parsed_dir, ch, args.text_only)
        chars = sum(len(i.get("text", "")) for i in items if i.get("type") == "text")
        total_chars += chars
        mark = "  ✅ already built" if ch in done else ""
        print(f"{ch}: {len(items)} blocks, ~{chars // 4} tokens  {dict(counts)}{mark}")
    mode = "text-only" if args.text_only else "full multimodal"
    print(f"\nMode: {mode}; knowledge base: {args.working_dir}; total ~{total_chars // 4} text tokens")

    if not args.run:
        print("\nDry run, no API calls made. Add --run to build.")
        return None

    # imported here so that a dry run works even without an API key
    from raganything import RAGAnything, RAGAnythingConfig
    from gemini_funcs import llm_func, vision_func, embedding_func, stats, cost_usd

    args.working_dir.mkdir(parents=True, exist_ok=True)
    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=str(args.working_dir),
            enable_image_processing=not args.text_only,
            enable_table_processing=not args.text_only,
            enable_equation_processing=not args.text_only,
        ),
        llm_model_func=llm_func,
        vision_model_func=vision_func,
        embedding_func=embedding_func,
        lightrag_kwargs=LIGHTRAG_KWARGS,
    )

    report = {"name": args.name, "text_only": args.text_only, "chapters": {}}
    t_all = time.time()
    for ch in args.chapters:
        if ch in done:
            continue
        items, _ = load_chapter(args.parsed_dir, ch, args.text_only)
        print(f"\n⏳ {ch}: building...")
        t0 = time.time()
        await rag.insert_content_list(items, file_path=f"Lay_{ch}")
        minutes = (time.time() - t0) / 60
        print(f"✅ {ch}: done in {minutes:.1f} min | {stats} | ~${cost_usd():.2f}")
        report["chapters"][ch] = round(minutes, 2)
        done.add(ch)
        done_file.write_text(json.dumps(sorted(done)), encoding="utf-8")

    await rag.finalize_storages()
    report.update(minutes_total=round((time.time() - t_all) / 60, 2), api=dict(stats), cost_usd=round(cost_usd(), 3))
    (args.working_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAll done. {stats} | estimated cost ~${cost_usd():.2f}")
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        asyncio.run(build(args))
    except FileNotFoundError as e:
        sys.exit(f"❌ {e}\nParse the PDFs in Colab first (colab/parse_mineru.ipynb), see README.")


if __name__ == "__main__":
    main()
