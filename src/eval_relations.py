# eval_relations.py — compares relations in a built graph with the ontology's own relations
# (the gold standard the extractor never saw): requires (prerequisites), uses, broader.
#
#   python src/eval_relations.py                                  -> rag_storage/ontology_norm_text
#   python src/eval_relations.py rag_storage/baseline_text
#   python src/eval_relations.py --review 50                      + CSV of 50 random "not in gold"
#                                                                   prerequisites for manual labelling
#   python src/eval_relations.py --score-review results/review_requires_ontology_norm_text.csv
#
# Only edges whose BOTH endpoints are ontology elements can be judged against the gold
# standard; edges involving other entities are counted but not scored.
#
# Views, because LightRAG stores the graph as undirected and the model mixes relation types:
#   * pair      — is the unordered pair {A, B} linked by an edge of that type?
#   * directed  — direction recovered from the edge description ("A requires an understanding of B").
#   * any type  — is the pair linked by any of requires/uses/broader (separates "wrong type" from "missing").
#   * any edge  — is the pair linked by any edge at all; the only view that also applies to the
#                 untyped baseline (`pair_recall_any_edge`).
# For prerequisites, precision is also reported against the transitive closure of the gold
# graph: the ontology lists direct prerequisites only, so "A requires C" via B is indirect, not wrong.
import argparse
import csv
import json
import random
import re
from pathlib import Path

import networkx as nx

from config import STORAGE_DIR, RESULTS_DIR
from inspect_kb import RELATION_TYPES
from ontology import gold_edges, load_ontology

GRAPH_FIELD_SEP = "<SEP>"          # LightRAG joins merged descriptions with this separator
EVALUATED = ["requires", "uses", "broader"]
GAP = r"(?:\s+[\w'-]+){0,6}?"       # up to six filler words, e.g. "requires an understanding of"

# Phrases that state "SOURCE <rel> TARGET" (forward) or "TARGET ... SOURCE" (backward), per relation.
FORWARD = {
    "requires": [r"requires", r"depends on", r"relies on", r"builds on", r"presupposes", r"needs"],
    "uses": [r"uses", r"relies on", r"applies", r"employs", r"is computed using", r"is based on"],
    "broader": [r"broader", r"is a (?:special|particular) case of", r"is a (?:narrower )?(?:kind|type|form) of",
                r"is an? (?:specific|particular) (?:kind|type) of", r"is a subclass of"],
}
BACKWARD = {   # phrases where the TARGET comes first: "B is a prerequisite for A" means A requires B
    "requires": [r"is (?:a )?prerequisite (?:for|of|to)", r"is required (?:for|by|to understand)",
                 r"is needed (?:for|to understand)"],
    "uses": [r"is used (?:in|by|for)"],
    "broader": [r"is a generalization of", r"generalizes", r"includes as a special case"],
}


def edge_types(data: dict) -> set:
    kws = {k.strip().lower().replace(" ", "_") for k in (data.get("keywords") or "").split(",") if k.strip()}
    return {t for t in RELATION_TYPES if t in kws}


def _states(text: str, first: set, second: set, phrases: list) -> bool:
    """Does the text say '<first> <phrase> [...] <second>' for any spelling of the two names?"""
    for a in first:
        for b in second:
            for ph in phrases:
                if re.search(rf"(?<!\w){re.escape(a)}\s+{ph}{GAP}\s+{re.escape(b)}(?!\w)", text):
                    return True
    return False


def direction(u: str, v: str, rel: str, description: str, u_aliases=(), v_aliases=()):
    """(source, target) as stated in the description, or None if it does not say (or says both).

    Aliases (ontology synonyms) are tried too: after normalisation a node may be called
    'Linear Independence' while its description still says 'Linearly Independent Set requires ...'.
    """
    text = " ".join((description or "").replace(GRAPH_FIELD_SEP, " ").split()).lower()
    names_u = {u.lower(), *(a.lower() for a in u_aliases)}
    names_v = {v.lower(), *(a.lower() for a in v_aliases)}
    fwd, bwd = FORWARD.get(rel, [rel]), BACKWARD.get(rel, [])
    uv = _states(text, names_u, names_v, fwd) or _states(text, names_v, names_u, bwd)
    vu = _states(text, names_v, names_u, fwd) or _states(text, names_u, names_v, bwd)
    if uv and not vu:
        return (u, v)
    if vu and not uv:
        return (v, u)
    return None


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"predicted": len(pred), "gold": len(gold), "correct": tp,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def collect(G: nx.Graph):
    """Typed edges between ontology elements: {rel: {"pairs": set, "directed": {(s, t): description}}}."""
    onto = load_ontology()
    index = onto.label_index()
    iri = {n: index[n.strip().lower()].iri for n in G.nodes if n.strip().lower() in index}
    aliases = {el.iri: el.all_labels for el in onto.elements}
    out = {rel: {"pairs": set(), "directed": {}, "no_direction": 0, "outside": 0, "edges": 0} for rel in EVALUATED}
    any_pairs = set()
    for u, v, d in G.edges(data=True):
        inside = u in iri and v in iri and iri[u] != iri[v]
        if inside:
            any_pairs.add(frozenset((iri[u], iri[v])))
        for rel in edge_types(d) & set(EVALUATED):
            o = out[rel]
            o["edges"] += 1
            if not inside:
                o["outside"] += 1
                continue
            o["pairs"].add(frozenset((iri[u], iri[v])))
            st = direction(u, v, rel, d.get("description"), aliases[iri[u]], aliases[iri[v]])
            if st is None:
                o["no_direction"] += 1
            else:
                o["directed"][(iri[st[0]], iri[st[1]])] = d.get("description") or ""
    return out, any_pairs


def evaluate(G: nx.Graph) -> dict:
    onto = load_ontology()
    label = {el.iri: el.pref_label for el in onto.elements}
    gold = gold_edges()
    found, any_pairs = collect(G)

    typed_pairs = set().union(*(found[r]["pairs"] for r in EVALUATED))
    gold_all_pairs = set().union(*({frozenset(e) for e in gold[r]} for r in EVALUATED))
    result = {"any_type": prf(typed_pairs, gold_all_pairs)}
    for rel in EVALUATED:
        f = found[rel]
        gold_dir = gold[rel]
        gold_pairs = {frozenset(e) for e in gold_dir}
        pred_dir = set(f["directed"])
        r = {
            "edges_of_this_type": f["edges"],
            "edges_outside_ontology": f["outside"],
            "pair": prf(f["pairs"], gold_pairs),
            "directed": prf(pred_dir, gold_dir),
            "direction_unknown": f["no_direction"],
            "reversed": len({(t, s) for s, t in pred_dir} & gold_dir - pred_dir),
            "gold_pairs_found_with_another_type": len((gold_pairs - f["pairs"]) & typed_pairs),
            "pair_recall_any_edge": round(len(any_pairs & gold_pairs) / len(gold_pairs), 3) if gold_pairs else 0,
            "correct_examples": sorted(f"{label[s]} -> {label[t]}" for s, t in pred_dir & gold_dir)[:20],
            "not_in_gold_examples": sorted(f"{label[s]} -> {label[t]}" for s, t in pred_dir - gold_dir)[:30],
            "missed_examples": sorted(f"{label[s]} -> {label[t]}" for s, t in gold_dir - pred_dir)[:30],
        }
        if rel == "requires":
            closure = set(nx.transitive_closure(nx.DiGraph(list(gold_dir)), reflexive=False).edges)
            r["directed_precision_vs_closure"] = round(len(pred_dir & closure) / len(pred_dir), 3) if pred_dir else 0
        result[rel] = r
    return result


def write_review(G: nx.Graph, n: int, path: Path, seed: int = 42) -> int:
    """CSV with a random sample of predicted prerequisites that are NOT in the gold standard."""
    onto = load_ontology()
    label = {el.iri: el.pref_label for el in onto.elements}
    gold = gold_edges()["requires"]
    closure = set(nx.transitive_closure(nx.DiGraph(list(gold)), reflexive=False).edges)
    found, _ = collect(G)
    candidates = sorted((e, desc) for e, desc in found["requires"]["directed"].items() if e not in gold)
    sample = random.Random(seed).sample(candidates, min(n, len(candidates)))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(["id", "source", "requires_target", "indirect_in_gold", "description",
                    "verdict (1 = correct prerequisite / 0 = wrong)", "comment"])
        for i, ((s, t), desc) in enumerate(sample, 1):
            text = " ".join(desc.replace(GRAPH_FIELD_SEP, " | ").split())
            w.writerow([i, label[s], label[t], "yes" if (s, t) in closure else "no", text[:500], "", ""])
    return len(candidates)


def score_review(path: Path, kb_result: dict) -> dict:
    """Estimated precision of prerequisites: gold matches + (share judged correct in the sample) x the rest."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    # Excel with a Ukrainian/European locale saves CSV with ';' instead of ','
    header = text.splitlines()[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    col = next(c for c in rows[0] if c.startswith("verdict"))
    judged = [r for r in rows if r[col].strip() in {"0", "1"}]
    if not judged:
        raise SystemExit(f"No verdicts (0/1) filled in {path}")
    share = sum(r[col].strip() == "1" for r in judged) / len(judged)
    d = kb_result["requires"]["directed"]
    not_gold = d["predicted"] - d["correct"]
    est = (d["correct"] + share * not_gold) / d["predicted"] if d["predicted"] else 0
    return {"judged": len(judged), "share_correct_among_not_in_gold": round(share, 3),
            "precision_strict": d["precision"], "precision_estimated": round(est, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate extracted relations against the ontology")
    p.add_argument("kb", nargs="?", type=Path, default=STORAGE_DIR / "ontology_norm_text")
    p.add_argument("--review", type=int, default=0, help="write a CSV sample of N 'not in gold' prerequisites")
    p.add_argument("--score-review", type=Path, help="labelled review CSV -> estimated precision")
    args = p.parse_args(argv)

    G = nx.read_graphml(args.kb / "graph_chunk_entity_relation.graphml")
    res = evaluate(G)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.score_review:
        s = score_review(args.score_review, res)
        res["requires"]["manual_review"] = s
        print(f"Manual review: {s['judged']} judged, {s['share_correct_among_not_in_gold']:.0%} of 'not in gold' "
              f"prerequisites are correct -> estimated precision {s['precision_estimated']:.0%} "
              f"(strict {s['precision_strict']:.0%})\n")

    out = RESULTS_DIR / f"relations_{args.kb.name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"=== Relations vs. ontology gold standard ({args.kb.name}) ===\n")
    print("| Relation | Edges of type | Pair P | Pair R | Directed P | Directed R | Directed F1 "
          "| Gold pairs linked by any edge |")
    print("|---|---|---|---|---|---|---|---|")
    for rel in EVALUATED:
        r = res[rel]
        print(f"| {rel} | {r['edges_of_this_type']} | {r['pair']['precision']:.0%} | {r['pair']['recall']:.0%} "
              f"| {r['directed']['precision']:.0%} | {r['directed']['recall']:.0%} | {r['directed']['f1']:.2f} "
              f"| {r['pair_recall_any_edge']:.0%} |")
    a = res["any_type"]
    print(f"| any of the three | {sum(res[r]['edges_of_this_type'] for r in EVALUATED)} | {a['precision']:.0%} "
          f"| {a['recall']:.0%} | | | | |")
    for rel in EVALUATED:
        r = res[rel]
        print(f"\n{rel}: gold {r['directed']['gold']}; predicted pairs between ontology elements "
              f"{r['pair']['predicted']} ({r['edges_outside_ontology']} edges involve other entities); "
              f"direction unknown {r['direction_unknown']}, reversed {r['reversed']}; "
              f"gold pairs found only with another type: {r['gold_pairs_found_with_another_type']}")
    rq = res["requires"]
    print(f"\nrequires: directed precision against the transitive closure of the gold graph: "
          f"{rq['directed_precision_vs_closure']:.0%}")
    print("\nCorrect prerequisites (sample):")
    for e in rq["correct_examples"][:10]:
        print(f"  {e}")
    print("\nPredicted prerequisites not in the gold standard (sample):")
    for e in rq["not_in_gold_examples"][:15]:
        print(f"  {e}")

    if args.review:
        path = RESULTS_DIR / f"review_requires_{args.kb.name}.csv"
        total = write_review(G, args.review, path)
        print(f"\nReview sample: {min(args.review, total)} of {total} 'not in gold' prerequisites -> {path}")
        print("Fill the verdict column with 1 (correct prerequisite) or 0 (wrong), then run:")
        print(f"  python src/eval_relations.py {args.kb} --score-review {path}")
    print(f"\nFull results: {out}")
    return res


if __name__ == "__main__":
    main()
