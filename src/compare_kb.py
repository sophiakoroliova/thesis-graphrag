# compare_kb.py — side-by-side table of graph metrics for several knowledge bases.
# Run inspect_kb.py for each knowledge base first (it writes results/graph_metrics_<name>.json).
#
#   python src/compare_kb.py baseline_text ontology_text
#   python src/compare_kb.py pilot_baseline_ch3_text pilot_ontology_ch3_text
import argparse
import json

from config import RESULTS_DIR

ROWS = [
    ("Entities", "nodes", "{}"),
    ("Relations", "edges", "{}"),
    ("Entity types", "entity_types", "{}"),
    ("Isolated nodes", "isolated_share", "{:.1%}"),
    ("Notation entities (Matrix A, T...)", "notation_entities", "{}"),
    ("Book-structure references (Theorem 8...)", "structural_entities", "{}"),
    ("Max degree (hub size)", "max_degree", "{}"),
    ("Entities matching the ontology", "entities_matching_ontology_share", "{:.1%}"),
    ("... excluding worked examples", "entities_matching_ontology_share_excl_examples", "{:.1%}"),
    ("Duplicate nodes of an ontology element", "ontology_duplicates", "{}"),
    ("Ontology elements covered (all 111)", "ontology_coverage", "{:.1%}"),
    ("Ontology elements of included chapters covered", "in_scope_coverage", "{:.1%}"),
    ("Relations typed (requires/uses/broader/...)", "relations_typed_share", "{:.1%}"),
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare graph metrics of knowledge bases")
    p.add_argument("names", nargs="+", help="knowledge base folder names, e.g. baseline_text ontology_text")
    args = p.parse_args(argv)
    metrics = [json.loads((RESULTS_DIR / f"graph_metrics_{n}.json").read_text(encoding="utf-8")) for n in args.names]

    header = "| Metric | " + " | ".join(args.names) + " |"
    print(header)
    print("|" + "---|" * (len(args.names) + 1))
    for label, key, fmt in ROWS:
        cells = [fmt.format(m[key]) if key in m else "—" for m in metrics]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
