# preprocess.py — prepares MinerU output (content_list) for insertion into RAG-Anything.
# No API calls, so it is easy to unit-test (tests/test_preprocess.py).
import json
import os
from collections import Counter
from pathlib import Path

# Page-layout blocks: the standard RAG-Anything pipeline drops them during parsing.
# We insert a pre-parsed content_list, so we drop them here in the same way.
LAYOUT_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote"}


def find_content_list(parsed_dir: Path, chapter: str) -> Path:
    """Finds *_content_list.json for a chapter (ignoring the _v2 variant)."""
    files = [p for p in (Path(parsed_dir) / f"lay_{chapter}").rglob("*_content_list.json")
             if not p.name.endswith("_v2.json")]
    if not files:
        raise FileNotFoundError(f"No content_list found for {chapter} in {parsed_dir}/lay_{chapter}")
    return files[0]


def multimodal_items(items, base_dir):
    """Standard RAG-Anything mode: drop layout blocks only, keep everything else as is."""
    out, counts = [], Counter()
    for it in items:
        t = it.get("type")
        if t in LAYOUT_TYPES:
            counts["dropped (layout)"] += 1
            continue
        it = dict(it)
        if it.get("img_path") and not os.path.isabs(it["img_path"]):
            it["img_path"] = os.path.join(str(base_dir), it["img_path"])  # RAG-Anything needs absolute paths
        out.append(it)
        counts[t] += 1
    return out, counts


def text_only_items(items):
    """Cost-saving mode: equations as LaTeX in the text, figures and tables as text.

    RAG-Anything then makes no separate LLM calls per equation or figure,
    and each equation stays next to the definition or theorem it belongs to.
    """
    out, counts = [], Counter()
    for it in items:
        t = it.get("type")
        page = it.get("page_idx", 0)
        if t in LAYOUT_TYPES:
            counts["dropped (layout)"] += 1
            continue
        if t == "text":
            text = (it.get("text") or "").strip()
            if text:
                item = {"type": "text", "text": text, "page_idx": page}
                if it.get("text_level"):
                    item["text_level"] = it["text_level"]
                out.append(item)
                counts["text"] += 1
        elif t == "equation":
            latex = (it.get("text") or "").strip()
            if latex:
                if not latex.startswith("$$"):
                    latex = f"$$\n{latex}\n$$"
                out.append({"type": "text", "text": latex, "page_idx": page})
                counts["equation -> LaTeX"] += 1
        elif t == "image":
            caption = " ".join(it.get("image_caption") or []).strip()
            if caption:
                out.append({"type": "text", "text": f"[Figure] {caption}", "page_idx": page})
                counts["image -> caption"] += 1
            else:
                counts["image without caption (skipped)"] += 1
        elif t == "table":
            caption = " ".join(it.get("table_caption") or []).strip()
            body = (it.get("table_body") or "").strip()
            head = f"[Table] {caption}" if caption else "[Table]"
            out.append({"type": "text", "text": "\n".join(x for x in [head, body] if x), "page_idx": page})
            counts["table -> text"] += 1
        else:
            text = (it.get("text") or "").strip()
            if text:
                out.append({"type": "text", "text": text, "page_idx": page})
                counts[f"{t} -> text"] += 1
    return out, counts


def load_chapter(parsed_dir: Path, chapter: str, text_only: bool):
    """Returns (blocks to insert, statistics) for one chapter."""
    path = find_content_list(parsed_dir, chapter)
    items = json.loads(path.read_text(encoding="utf-8"))
    if text_only:
        return text_only_items(items)
    return multimodal_items(items, path.parent.resolve())
