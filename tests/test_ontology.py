# Offline tests for the ontology-guided method (no API calls).
#   pytest tests/test_ontology.py
import copy

import pytest

from ontology import ENTITY_CLASSES, gold_relations, load_ontology, title_case
from ontology_prompt import DOMAIN_EXAMPLE, apply_ontology_guidance, build_ontology_section


def test_ontology_loads_all_elements():
    onto = load_ontology()
    assert len(onto.elements) == 111
    assert {e.cls for e in onto.elements} == {"Concept", "Theorem", "Method"}


def test_label_index_maps_synonyms_to_canonical_names():
    index = load_ontology().label_index()
    assert index["bases"].pref_label == "basis"
    assert index["kernel"].pref_label == "kernel of a linear transformation"


def test_title_case():
    assert title_case("kernel of a linear transformation") == "Kernel of a Linear Transformation"
    assert title_case("LU factorization") == "LU Factorization"


def test_no_gold_relation_leaks_into_prompt():
    """The ontology's own relations are the evaluation gold standard and must not be in the prompt."""
    text = (build_ontology_section(load_ontology()) + DOMAIN_EXAMPLE).lower()
    leaks = [t for t in gold_relations()
             if f"{t[0].lower()} {t[1]} {t[2].lower()}" in text
             or f"{title_case(t[0]).lower()} {t[1]} {title_case(t[2]).lower()}" in text]
    assert leaks == []


def test_prompt_renders_with_installed_lightrag():
    prompt_mod = pytest.importorskip("lightrag.prompt")
    prompts = copy.deepcopy(prompt_mod.PROMPTS)
    if "---Examples---" not in prompts["entity_extraction_system_prompt"]:
        pytest.skip("installed LightRAG is not 1.4.x")
    from ontology_prompt import render_system_prompt
    addon = apply_ontology_guidance(prompts)
    assert addon["entity_types"] == ENTITY_CLASSES
    text = render_system_prompt(prompts, addon["entity_types"])   # raises if braces are unescaped
    assert "---Domain Ontology---" in text and "Orthogonal Set" in text
    apply_ontology_guidance(prompts)                              # idempotent
    assert prompts["entity_extraction_system_prompt"].count("---Domain Ontology---") == 1


def test_ontology_alignment_metric():
    import networkx as nx
    from inspect_kb import ontology_alignment
    G = nx.Graph()
    G.add_nodes_from(["Basis", "Bases", "Kernel", "Matrix A"])
    m = ontology_alignment(G, load_ontology().label_index())
    assert m["entities_matching_ontology"] == 3        # Basis, Bases (synonym), Kernel (synonym)
    assert m["ontology_elements_covered"] == 2         # basis, kernel of a linear transformation


def test_elements_know_their_chapter():
    chapters = {e.pref_label: e.chapter for e in load_ontology().elements}
    assert chapters["determinant"] == 3
    assert chapters["basis"] in (2, 4)
    ch3 = [e for e in load_ontology().elements if e.chapter == 3]
    assert len(ch3) == 8


def test_relation_typing_metric():
    import networkx as nx
    from inspect_kb import relation_typing
    G = nx.Graph()
    G.add_edge("Basis", "Span", keywords="definition,requires")      # LightRAG stores keywords sorted
    G.add_edge("Cramer's Rule", "Determinant", keywords="uses")
    G.add_edge("Example", "Determinant", keywords="computation,example of")
    G.add_edge("Matrix A", "Vector", keywords="vector properties")
    m = relation_typing(G)
    assert m["relations_typed"] == 3 and m["relation_types"]["untyped"] == 1
    assert m["relation_types"]["example_of"] == 1
