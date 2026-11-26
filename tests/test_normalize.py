# Offline tests for the normalisation plan (no storage, no API calls).
import networkx as nx

from normalize_kb import default_output, plan_normalization
from ontology import load_ontology


def make_graph():
    G = nx.Graph()
    G.add_node("Basis", entity_type="concept")
    G.add_node("Bases", entity_type="concept")              # hiddenLabel of basis
    G.add_node("Kernel", entity_type="concept")             # altLabel of 'kernel of a linear transformation'
    G.add_node("Cramer's Rule", entity_type="concept")      # ontology says Theorem
    G.add_node("Theorem 8", entity_type="theorem")          # numbered reference
    G.add_node("Area of a Parallelogram", entity_type="concept")  # not in the ontology
    G.add_edge("Basis", "Bases")
    return G


def test_plan_merges_synonyms_into_canonical_names():
    plan = plan_normalization(make_graph(), load_ontology().label_index())
    merges = {m["target"]: m["sources"] for m in plan["merges"]}
    assert merges["Basis"] == ["Bases"]
    assert merges["Kernel of a Linear Transformation"] == ["Kernel"]


def test_plan_fixes_types_and_removes_numbered_references():
    plan = plan_normalization(make_graph(), load_ontology().label_index())
    assert {"entity": "Cramer's Rule", "from": "concept", "to": "theorem"} in plan["retypes"]
    assert plan["deletions"] == ["Theorem 8"]


def test_plan_leaves_entities_outside_the_ontology_alone():
    plan = plan_normalization(make_graph(), load_ontology().label_index())
    touched = {n for m in plan["merges"] for n in m["sources"] + [m["target"]]}
    touched |= {r["entity"] for r in plan["retypes"]} | set(plan["deletions"])
    assert "Area of a Parallelogram" not in touched


def test_default_output_name():
    from pathlib import Path
    assert default_output(Path("rag_storage/ontology_text")).name == "ontology_norm_text"
