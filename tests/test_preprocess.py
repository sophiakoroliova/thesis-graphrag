# Offline tests: no API calls, no data needed, run in under a second.
#   pytest tests/test_preprocess.py
import networkx as nx

import config
from inspect_kb import compute_metrics
from preprocess import multimodal_items, text_only_items

SAMPLE = [
    {"type": "header", "text": "56 CHAPTER 1 Linear Equations in Linear Algebra", "page_idx": 0},
    {"type": "page_number", "text": "56", "page_idx": 0},
    {"type": "text", "text": "1.7 LINEAR INDEPENDENCE", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "An indexed set of vectors is linearly independent if", "page_idx": 0},
    {"type": "equation", "text": "$$\nx_1 v_1 + \\dots + x_p v_p = 0\n$$", "page_idx": 0},
    {"type": "equation", "text": "c_1 v_1 = 0", "page_idx": 0},
    {"type": "image", "img_path": "images/a.jpg", "image_caption": ["FIGURE 1 Linearly dependent"], "page_idx": 1},
    {"type": "image", "img_path": "images/b.jpg", "page_idx": 1},
    {"type": "table", "table_caption": ["Table 1"], "table_body": "<table></table>", "page_idx": 1},
]


def test_text_only_drops_layout_blocks():
    out, counts = text_only_items(SAMPLE)
    texts = " ".join(i["text"] for i in out)
    assert "CHAPTER 1" not in texts and counts["dropped (layout)"] == 2


def test_text_only_keeps_equations_as_latex():
    out, _ = text_only_items(SAMPLE)
    eqs = [i["text"] for i in out if "v_1" in i["text"]]
    assert len(eqs) == 2 and all(e.startswith("$$") for e in eqs)


def test_text_only_images_become_captions_or_are_skipped():
    out, counts = text_only_items(SAMPLE)
    assert any(i["text"] == "[Figure] FIGURE 1 Linearly dependent" for i in out)
    assert counts["image without caption (skipped)"] == 1


def test_text_only_output_is_text_only():
    out, _ = text_only_items(SAMPLE)
    assert {i["type"] for i in out} == {"text"}
    assert out[0].get("text_level") == 1  # headings keep their level


def test_multimodal_keeps_equations_and_makes_image_paths_absolute():
    out, counts = multimodal_items(SAMPLE, "/base/dir")
    assert counts["equation"] == 2 and counts["image"] == 2
    assert all(i["img_path"].startswith("/base/dir") for i in out if i["type"] == "image")
    assert all(i["type"] not in {"header", "page_number"} for i in out)


def test_chapter_ranges_cover_238_pages():
    pages = sum(e - s + 1 for ranges in config.CHAPTER_RANGES.values() for s, e in ranges)
    assert pages == 238


def test_graph_metrics_detect_baseline_defects():
    G = nx.Graph()
    G.add_node("Matrix A", entity_type="data")
    G.add_node("Linear Independence", entity_type="concept")
    G.add_node("Theorem 8", entity_type="concept")
    G.add_node("T", entity_type="concept")
    G.add_node("Study Guide", entity_type="artifact")
    G.add_edge("Matrix A", "Linear Independence")
    G.add_edge("Matrix A", "Theorem 8")
    m = compute_metrics(G)
    assert m["notation_entities"] == 2          # Matrix A, T
    assert m["structural_entities"] == 1        # Theorem 8
    assert m["isolated"] == 2                   # T, Study Guide
    assert m["top_degree"][0][0] == "Matrix A"
