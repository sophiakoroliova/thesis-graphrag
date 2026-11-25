# Smoke test: the whole pipeline on the smallest chapter (chapter 3) with real Gemini calls.
# Costs about $0.15 and a few minutes, so it is skipped by default.
#
#   pytest tests/test_smoke.py --smoke
#
# Checks that the knowledge base builds without errors, the graph is not empty,
# and the system answers a question.
import asyncio
import json
import os
import shutil

import networkx as nx
import pytest
from dotenv import load_dotenv

import config

load_dotenv(config.ROOT / ".env")

SMOKE_DIR = config.STORAGE_DIR / "_smoke_test"


@pytest.fixture(scope="module")
def built_kb():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("no GEMINI_API_KEY in .env")
    if not list((config.PARSED_DIR / "lay_ch3").rglob("*_content_list.json")):
        pytest.skip("no parsed output for chapter 3")
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)

    import build_kb
    args = build_kb.parse_args(["--run", "--text-only", "--chapters", "ch3", "--working-dir", str(SMOKE_DIR)])
    report = asyncio.run(build_kb.build(args))
    yield SMOKE_DIR, report
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)


@pytest.mark.smoke
def test_document_processed(built_kb):
    kb, report = built_kb
    assert "ch3" in report["chapters"]
    status = json.loads((kb / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    assert all(d.get("status") == "processed" for d in status.values())


@pytest.mark.smoke
def test_graph_not_empty(built_kb):
    kb, _ = built_kb
    G = nx.read_graphml(kb / "graph_chunk_entity_relation.graphml")
    assert G.number_of_nodes() > 20 and G.number_of_edges() > 20


@pytest.mark.smoke
def test_answers_question(built_kb):
    kb, _ = built_kb
    import ask_kb

    async def go():
        rag = await ask_kb.open_kb(kb)
        answer, _ = await ask_kb.ask(rag, "How is the determinant of a triangular matrix computed?")
        await rag.finalize_storages()
        return answer

    answer = asyncio.run(go())
    assert isinstance(answer, str) and len(answer) > 50
    assert "diagonal" in answer.lower()
