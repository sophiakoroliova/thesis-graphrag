# run_all.py — the whole local pipeline in one command.
#
#   python run_all.py --text-only                 baseline in text-only mode (as in the experiment)
#   python run_all.py --text-only --chapters ch3  quick run on a single chapter
#   python run_all.py --text-only --ask           + sample questions at the end
#
# Steps:
#   1. cut       — cut corpus and chapters from data/raw/Lay.pdf (skipped if already done)
#   2. parsed?   — check that MinerU output exists (produced by colab/parse_mineru.ipynb)
#   3. build     — build the knowledge base
#   4. inspect   — compute graph metrics -> results/
#   5. ask       — (optional) sample questions
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import config  # noqa: E402


def step(title):
    print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")


def run(script, *args):
    cmd = [sys.executable, str(SRC / script), *map(str, args)]
    print("$", " ".join(cmd[1:]))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        sys.exit(f"❌ Step {script} failed (exit code {res.returncode})")


def main():
    p = argparse.ArgumentParser(description="Full pipeline: PDF -> knowledge base -> metrics")
    p.add_argument("--text-only", action="store_true", help="text-only mode (equations as LaTeX, figures as captions)")
    p.add_argument("--chapters", nargs="+", choices=config.CHAPTERS, default=config.CHAPTERS)
    p.add_argument("--name", default="baseline", help="configuration name")
    p.add_argument("--ask", action="store_true", help="ask sample questions at the end")
    args = p.parse_args()

    step("1. Cut PDF")
    chapter_pdfs = [config.PDF_DIR / "chapters" / f"lay_{ch}.pdf" for ch in args.chapters]
    if all(p.exists() for p in chapter_pdfs):
        print("Chapter PDFs already exist, skipping")
    elif config.RAW_PDF.exists():
        run("cut_pdf.py")
    else:
        print(f"⚠️  {config.RAW_PDF.relative_to(ROOT)} not found; skipping (only needed for parsing)")

    step("2. Check parsed corpus")
    missing = [ch for ch in args.chapters if not list((config.PARSED_DIR / f"lay_{ch}").rglob("*_content_list.json"))]
    if missing:
        sys.exit(f"❌ No MinerU output for: {', '.join(missing)}.\n"
                 f"   Run colab/parse_mineru.ipynb and put the hybrid_full folder into "
                 f"{config.PARSED_DIR.relative_to(ROOT)} (see README, step 3).")
    print("✅ parsed output found for all chapters")

    flags = ["--text-only"] if args.text_only else []
    step("3. Build knowledge base")
    run("build_kb.py", "--run", "--name", args.name, "--chapters", *args.chapters, *flags)

    kb = config.STORAGE_DIR / (f"{args.name}_text" if args.text_only else args.name)
    step("4. Graph metrics")
    run("inspect_kb.py", kb)

    if args.ask:
        step("5. Sample questions")
        run("ask_kb.py", "--kb", kb)

    print("\n✅ Pipeline finished")


if __name__ == "__main__":
    main()
