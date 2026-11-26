# Offline tests for relation evaluation against the ontology gold standard.
import networkx as nx

from eval_relations import direction, evaluate, prf
from ontology import gold_edges, load_ontology


def labels():
    onto = load_ontology()
    return {el.iri: el.pref_label for el in onto.elements}


def test_direction_from_description():
    assert direction("Basis", "Span", "requires", "Basis requires Span, because...") == ("Basis", "Span")
    assert direction("Span", "Basis", "requires", "x<SEP>Basis requires Span") == ("Basis", "Span")
    assert direction("Basis", "Span", "requires", "Basis is related to Span") is None
    assert direction("Linear Independence", "Span", "requires", "Linearly Independent Set requires Span",
                     u_aliases=["linearly independent set"]) == ("Linear Independence", "Span")


def test_prf():
    assert prf({1, 2}, {2, 3}) == {"predicted": 2, "gold": 2, "correct": 1,
                                   "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_evaluate_scores_a_gold_prerequisite_and_a_reversed_one():
    lab = labels()
    s, t = sorted(gold_edges()["requires"])[0]          # any gold prerequisite s requires t
    a, b = lab[s].title(), lab[t].title()
    G = nx.Graph()
    G.add_edge(a, b, keywords="definition,requires", description=f"{a} requires {b}, because ...")
    res = evaluate(G)["requires"]
    assert res["directed"]["correct"] == 1 and res["directed"]["precision"] == 1.0
    G2 = nx.Graph()
    G2.add_edge(a, b, keywords="requires", description=f"{b} requires {a}")
    res2 = evaluate(G2)["requires"]
    assert res2["directed"]["correct"] == 0 and res2["reversed"] == 1
    assert res2["pair"]["correct"] == 1                  # the pair itself is right


def test_untyped_edges_only_count_for_any_edge_recall():
    lab = labels()
    s, t = sorted(gold_edges()["requires"])[0]
    G = nx.Graph()
    G.add_edge(lab[s], lab[t], keywords="vector properties", description="...")
    res = evaluate(G)["requires"]
    assert res["pair"]["predicted"] == 0 and res["pair_recall_any_edge"] > 0


def test_direction_understands_paraphrases():
    assert direction("Basis", "Linear Independence", "requires",
                     "Basis requires an understanding of Linear Independence") == ("Basis", "Linear Independence")
    assert direction("Basis", "Span", "requires", "Span is a prerequisite for Basis") == ("Basis", "Span")
    assert direction("Square Matrix", "Matrix", "broader",
                     "Square Matrix is a special case of Matrix") == ("Square Matrix", "Matrix")
    assert direction("Cramer's Rule", "Determinant", "uses",
                     "Determinant is used in Cramer's Rule") == ("Cramer's Rule", "Determinant")


def test_review_csv_and_scoring(tmp_path):
    from eval_relations import score_review, write_review
    lab = labels()
    gold = sorted(gold_edges()["requires"])
    G = nx.Graph()
    s, t = gold[0]
    G.add_edge(lab[s], lab[t], keywords="requires", description=f"{lab[s]} requires {lab[t]}")
    # a pair that is not a gold prerequisite (reverse direction of another gold edge)
    s2, t2 = gold[1]
    G.add_edge(lab[t2], lab[s2], keywords="requires", description=f"{lab[t2]} requires {lab[s2]}")
    path = tmp_path / "review.csv"
    assert write_review(G, 10, path) == 1
    text = path.read_text(encoding="utf-8-sig").replace(",,", ",1,", 1)   # mark the one row as correct
    path.write_text(text, encoding="utf-8-sig")
    est = score_review(path, evaluate(G))
    assert est["share_correct_among_not_in_gold"] == 1.0 and est["precision_estimated"] == 1.0


def test_score_review_accepts_semicolon_csv(tmp_path):
    from eval_relations import score_review
    lab = labels()
    s, t = sorted(gold_edges()["requires"])[1]
    G = nx.Graph()
    G.add_edge(lab[t], lab[s], keywords="requires", description=f"{lab[t]} requires {lab[s]}")
    path = tmp_path / "review.csv"
    path.write_text("id;source;requires_target;indirect_in_gold;description;"
                    "verdict (1 = correct prerequisite / 0 = wrong);comment\n1;a;b;no;x;0;\n", encoding="utf-8-sig")
    assert score_review(path, evaluate(G))["share_correct_among_not_in_gold"] == 0.0
