# Offline tests for the answer evaluation (no API calls).
import itertools
import json

import pytest

import config
import eval_answers as ea


# ---------- question set ----------
def test_question_set():
    qs = ea.load_questions()
    assert len(qs) == 50 and len({q["id"] for q in qs}) == 50
    counts = {t: sum(q["type"] == t for q in qs) for t in ea.TYPES}
    assert counts == {"definition": 15, "relation": 15, "prerequisite": 10, "cross-chapter": 10}
    # every cited page lies inside the corpus (book page = PDF page - 17)
    book_ranges = [(a - 17, b - 17) for ranges in config.CHAPTER_RANGES.values() for a, b in ranges]
    for q in qs:
        assert q["question"].strip() and q["reference"].strip() and q["pages"]
        for p in q["pages"]:
            assert any(a <= p <= b for a, b in book_ranges), (q["id"], p)


def test_pilot_limit_covers_all_types():
    qs = ea.load_questions(limit=8)
    assert len(qs) == 8 and {q["type"] for q in qs} == set(ea.TYPES)


# ---------- metrics ----------
def test_claim_metrics():
    m = ea.claim_metrics([1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 1])
    assert m["fc_precision"] == 1.0 and m["fc_recall"] == pytest.approx(0.6667, abs=1e-4)
    assert m["fc_f1"] == pytest.approx(0.8, abs=1e-4) and m["faithfulness"] == 0.75
    # extra correct details lower precision but are not errors; contradictions are counted separately
    m = ea.claim_metrics([1, 0, 0, -1], [1, 1], [1, 1, 1, -1])
    assert m["fc_precision"] == 0.25 and m["fc_contradicted"] == 0.25 and m["fc_recall"] == 1.0
    assert m["faithfulness"] == 0.75 and m["context_contradicted"] == 0.25


def test_verdict_parsing():
    assert [ea._verdict(x) for x in (1, "1", 0, "0", -1, "-1", True, "x", None, 5)] == [1, 1, 0, 0, -1, -1, 1, 0, 0, 1]
    empty = ea.claim_metrics([], [1, 0], [])
    assert empty["fc_precision"] == 0 and empty["fc_f1"] == 0 and empty["faithfulness"] == 0


def _brute_force_p(x, y):
    """Exact two-sided p of the signed-rank test by enumerating all sign assignments."""
    d = [b - a for a, b in zip(x, y) if b != a]
    absd = [abs(v) for v in d]
    ranks = [sum(a < v for a in absd) + (sum(a == v for a in absd) + 1) / 2 for v in absd]
    w = sum(r for r, v in zip(ranks, d) if v > 0)
    lo = hi = 0
    for signs in itertools.product([0, 1], repeat=len(d)):
        s = sum(r for r, b in zip(ranks, signs) if b)
        lo += s <= w
        hi += s >= w
    return min(1.0, 2 * min(lo, hi) / 2 ** len(d))


def test_wilcoxon_exact_matches_brute_force_with_ties():
    x, y = [3, 3, 4, 2, 5, 1, 4, 3, 2, 5], [4, 5, 4, 3, 5, 3, 5, 4, 4, 4]
    res = ea.wilcoxon(x, y)
    assert res["method"] == "exact" and res["n"] == 8
    assert res["p"] == pytest.approx(_brute_force_p(x, y), abs=1e-4)
    assert res["w_plus"] == 33 and res["w_minus"] == 3


def test_wilcoxon_known_values():
    # reference values computed with scipy.stats.wilcoxon (zero_method="wilcox", no correction)
    assert ea.wilcoxon([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 3, 7, 9, 5, 10, 12])["p"] == pytest.approx(0.0469, abs=1e-4)
    x = list(range(30))
    y = [v + (1 if v % 3 else -1) * (v % 5 + 1) for v in x]
    res = ea.wilcoxon(x, y)
    assert res["method"] == "normal approximation" and res["p"] == pytest.approx(0.1093, abs=1e-4)


def test_wilcoxon_no_difference():
    assert ea.wilcoxon([1, 2, 3], [1, 2, 3])["p"] == 1.0


def test_percentile():
    assert ea.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == pytest.approx(9.1)


# ---------- retrieved context ----------
def test_context_text():
    data = {"data": {"entities": [{"entity_name": "Basis", "entity_type": "concept", "description": "a<SEP>b"}],
                     "relationships": [{"src_id": "Basis", "tgt_id": "Span", "description": "Basis requires Span"}],
                     "chunks": [{"content": "A basis for H is ..."}]}}
    text = ea.context_text(data)
    assert "Basis (concept): a | b" in text and "Basis -- Span" in text and "A basis for H" in text
    assert ea.context_counts(data) == {"entities": 1, "relationships": 1, "chunks": 1}


# ---------- manual check of the judge ----------
def test_judge_review_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "RESULTS_DIR", tmp_path)
    qs = ea.load_questions()
    for kb, score in (("kb_alpha", 3), ("kb_beta", 5)):
        for q in qs[:12] + qs[40:46]:
            ea.append_jsonl(tmp_path / f"answers_judged_{kb}.jsonl",
                            {"id": q["id"], "type": q["type"], "question": q["question"],
                             "answer": "some answer", "judge_score": score})
    path = tmp_path / "judge_review.csv"
    assert ea.write_judge_review(["kb_alpha", "kb_beta"], 10, path) == 10
    text = path.read_text(encoding="utf-8-sig")
    assert "kb_alpha" not in text and "kb_beta" not in text      # blind: no knowledge base names in the CSV
    key = json.loads(path.with_suffix(".key.json").read_text(encoding="utf-8"))
    # the author agrees with the judge on every item
    lines = text.splitlines()
    import csv
    rows = list(csv.reader(lines))
    for r in rows[1:]:
        r[4] = str(key[r[0]]["judge_score"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    res = ea.score_judge_review(path)
    assert res["judged"] == 10 and res["exact_agreement"] == 1.0 and res["agreement_within_1"] == 1.0


def test_parse_json_with_unescaped_latex():
    raw = '{"claims": ["The vector $\\mathbf{y} = c_1 \\mathbf{v}_1$ and \\lambda is a scalar", "x \\"quoted\\""]}'
    raw_bad = raw.replace("\\\\", "\\")       # what the model sometimes returns: single backslashes
    out = ea.parse_json(raw_bad)
    assert out["claims"][0].startswith("The vector $\\mathbf{y}") and "\\lambda" in out["claims"][0]
    assert ea.parse_json('{"verdicts": [1, 0, -1]}') == {"verdicts": [1, 0, -1]}
    # \frac and \beta begin with valid JSON escapes (\f, \b) and must not become control characters
    assert ea.parse_json(r'{"c": ["$\frac{1}{2}$ and \beta"]}') == {"c": ["$\\frac{1}{2}$ and \\beta"]}
    assert ea.parse_json(r'{"c": ["$\\frac{1}{2}$"]}') == {"c": ["$\\frac{1}{2}$"]}   # already escaped


# ---------- graph-only ablation ----------
def test_drop_text_chunks_replaces_chunk_selection(monkeypatch):
    import asyncio
    op = pytest.importorskip("lightrag.operate")
    monkeypatch.setattr(op, "process_chunks_unified", op.process_chunks_unified)   # restored after the test
    ea.drop_text_chunks()
    assert asyncio.run(op.process_chunks_unified("q", [{"content": "x"}], None, None)) == []


def test_judge_without_faithfulness(monkeypatch):
    import asyncio
    calls = []

    async def fake_judge(prompt):
        calls.append(prompt)
        if prompt.startswith("The TEXT answers"):
            return {"claims": ["c1", "c2"]}
        if "CLAIMS:" in prompt:
            return {"verdicts": [1, 1]}
        return {"score": 5, "reason": "ok"}
    monkeypatch.setattr(ea, "judge_json", fake_judge)
    q = ea.load_questions()[0]
    row = asyncio.run(ea.judge_one(q, {"answer": "a", "context": "CONTEXT-TEXT"}, ["r1", "r2"], faithfulness=False))
    assert row["faithfulness"] is None and row["fc_recall"] == 1.0 and row["judge_score"] == 5
    assert not any("CONTEXT-TEXT" in c for c in calls)        # the context was never sent to the judge
