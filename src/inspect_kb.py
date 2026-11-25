# inspect_kb.py — structural metrics of a built graph (no API calls).
#
#   python src/inspect_kb.py                               -> rag_storage/baseline_text
#   python src/inspect_kb.py rag_storage/ontology_text
#   python src/inspect_kb.py rag_storage/baseline_text --quiet
#
# Metrics are saved to results/graph_metrics_<kb name>.json and computed the same way
# for the baseline and the ontology-guided method, so they can be compared directly.
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx

from config import STORAGE_DIR, RESULTS_DIR

# Heuristics for typical baseline defects (see the 2026-11-19 analysis):
# variable symbols extracted as entities: "Matrix A", "Vector V1", "Basis B", "T", "H"
NOTATION_RE = re.compile(r"^(?:[A-Za-z]+ )?[A-Z]\d*$")
# references to book structure instead of content: "Theorem 8", "Example 3", "Exercise 12"
STRUCTURAL_RE = re.compile(r"^(?:Theorem|Example|Exercise|Figure|Section|Table)\s+\d+", re.I)


def compute_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    nodes = list(G.nodes)
    types = Counter(G.nodes[v].get("entity_type", "?") for v in nodes)
    isolated = [v for v in nodes if G.degree(v) == 0]
    notation = [v for v in nodes if NOTATION_RE.match(v)]
    structural = [v for v in nodes if STRUCTURAL_RE.match(v)]
    top = sorted(G.degree, key=lambda x: -x[1])[:25]
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "entity_types": len(types),
        "types": dict(types.most_common()),
        "isolated": len(isolated),
        "isolated_share": round(len(isolated) / n, 3) if n else 0,
        "notation_entities": len(notation),
        "notation_examples": notation[:15],
        "structural_entities": len(structural),
        "structural_examples": structural[:15],
        "max_degree": top[0][1] if top else 0,
        "top_degree": [[v, d, G.nodes[v].get("entity_type", "?")] for v, d in top],
    }


def doc_status(kb: Path) -> dict:
    path = kb / "kv_store_doc_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv=None):
    p = argparse.ArgumentParser(description="Knowledge graph metrics")
    p.add_argument("kb", nargs="?", type=Path, default=STORAGE_DIR / "baseline_text")
    p.add_argument("--quiet", action="store_true", help="only save the JSON")
    args = p.parse_args(argv)

    G = nx.read_graphml(args.kb / "graph_chunk_entity_relation.graphml")
    m = compute_metrics(G)
    status = doc_status(args.kb)
    m["documents"] = {d.get("file_path", k): d.get("status") for k, d in status.items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"graph_metrics_{args.kb.name}.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")

    if not args.quiet:
        print("=== DOCUMENTS ===")
        for name, st in m["documents"].items():
            print(f"  {name:12} {st}")
        print(f"\n=== GRAPH ({args.kb.name}) ===")
        print(f"Entities: {m['nodes']}, relations: {m['edges']}")
        print(f"Isolated nodes: {m['isolated']} ({m['isolated_share']:.0%})")
        print(f"Notation entities (Matrix A, T, V...): {m['notation_entities']}")
        print(f"Book-structure references (Theorem 8, Example 3...): {m['structural_entities']}")
        print(f"\nEntity types ({m['entity_types']}):")
        for t, c in list(m["types"].items())[:15]:
            print(f"  {t:25} {c}")
        print("\nMost connected entities:")
        for v, d, t in m["top_degree"]:
            print(f"  {d:4}  {v}  [{t}]")
    print(f"\nMetrics saved to: {out}")
    return m


if __name__ == "__main__":
    main()
