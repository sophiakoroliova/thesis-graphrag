# normalize_kb.py — post-processing step of the proposed method: name normalisation.
#
# Works on a COPY of an ontology-guided knowledge base and uses LightRAG's own editing API
# (amerge_entities, aedit_entity, adelete_by_entity), so the graph, the vector indexes and
# the chunk-tracking stores stay consistent and the result can be queried like any other KB.
#
#   1. synonyms -> canonical name: all entities whose name is a prefLabel/altLabel/hiddenLabel of the
#      same ontology element are merged into one entity named after its prefLabel (e.g. Bases -> Basis);
#   2. type alignment: such entities get the class of their ontology element (Concept/Theorem/Method);
#   3. numbered references that slipped through ("Theorem 8", "Example 3") are removed: they are
#      ambiguous across chapters and are not knowledge elements in the ontology.
#
# Only names and types from the ontology are used — never its relations.
# Entities outside the ontology are left untouched.
#
#   python src/normalize_kb.py                                  dry run: shows the plan, no changes
#   python src/normalize_kb.py --run                            rag_storage/ontology_text -> ontology_norm_text
#   python src/normalize_kb.py rag_storage/pilot_ontology_ch3_text --run
#
# Re-embedding merged entities costs a few embedding calls (well under one cent).
import argparse
import asyncio
import json
import shutil
from collections import defaultdict
from pathlib import Path

import networkx as nx

from config import STORAGE_DIR, RESULTS_DIR
from inspect_kb import STRUCTURAL_RE
from ontology import load_ontology, title_case


def default_output(kb: Path) -> Path:
    name = kb.name[:-5] + "_norm_text" if kb.name.endswith("_text") else kb.name + "_norm"
    return kb.parent / name


def plan_normalization(G: nx.Graph, label_index: dict, drop_structural: bool = True) -> dict:
    """Decides what to merge, retype and delete. Pure function: no storage access, no API calls."""
    groups = defaultdict(list)                       # ontology element IRI -> graph nodes
    elements = {}
    for node in G.nodes:
        el = label_index.get(node.strip().lower())
        if el is not None:
            groups[el.iri].append(node)
            elements[el.iri] = el

    merges, retypes = [], []
    for iri, nodes in groups.items():
        el = elements[iri]
        target = title_case(el.pref_label)
        wanted_type = el.cls.lower()
        sources = sorted(n for n in nodes if n != target)
        if sources:
            merges.append({"target": target, "sources": sources, "entity_type": wanted_type})
        elif G.nodes[target].get("entity_type", "").lower() != wanted_type:
            retypes.append({"entity": target, "from": G.nodes[target].get("entity_type"), "to": wanted_type})

    deletions = sorted(n for n in G.nodes if STRUCTURAL_RE.match(n)) if drop_structural else []
    return {"merges": merges, "retypes": retypes, "deletions": deletions}


def print_plan(plan: dict):
    print(f"Merges into canonical names: {len(plan['merges'])}")
    for m in plan["merges"]:
        print(f"  {', '.join(m['sources'])}  ->  {m['target']}  [{m['entity_type']}]")
    print(f"\nType corrections: {len(plan['retypes'])}")
    for r in plan["retypes"]:
        print(f"  {r['entity']}: {r['from']} -> {r['to']}")
    print(f"\nNumbered references to remove: {len(plan['deletions'])}")
    for d in plan["deletions"]:
        print(f"  {d}")


async def apply_plan(kb: Path, plan: dict):
    from ask_kb import open_kb

    rag = await open_kb(kb)
    lr = rag.lightrag
    for m in plan["merges"]:
        await lr.amerge_entities(source_entities=m["sources"], target_entity=m["target"],
                                 target_entity_data={"entity_type": m["entity_type"]})
    for r in plan["retypes"]:
        await lr.aedit_entity(r["entity"], {"entity_type": r["to"]})
    for d in plan["deletions"]:
        await lr.adelete_by_entity(d)
    await rag.finalize_storages()


def main(argv=None):
    p = argparse.ArgumentParser(description="Normalise entity names of an ontology-guided knowledge base")
    p.add_argument("kb", nargs="?", type=Path, default=STORAGE_DIR / "ontology_text")
    p.add_argument("--out", type=Path, default=None, help="output KB folder (default: <name>_norm_text)")
    p.add_argument("--run", action="store_true", help="apply the plan (without it: dry run)")
    p.add_argument("--keep-structural", action="store_true", help="do not remove 'Theorem 8'-style entities")
    p.add_argument("--force", action="store_true", help="overwrite the output folder if it exists")
    args = p.parse_args(argv)
    out = args.out or default_output(args.kb)

    G = nx.read_graphml(args.kb / "graph_chunk_entity_relation.graphml")
    plan = plan_normalization(G, load_ontology().label_index(), drop_structural=not args.keep_structural)
    print_plan(plan)

    if not args.run:
        print(f"\nDry run, nothing changed. Add --run to write the normalised KB to {out}")
        return plan

    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} already exists; use --force to overwrite")
        shutil.rmtree(out)
    shutil.copytree(args.kb, out)
    print(f"\nCopied {args.kb.name} -> {out.name}; applying...")
    asyncio.run(apply_plan(out, plan))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = RESULTS_DIR / f"normalization_{out.name}.json"
    report.write_text(json.dumps({"source": args.kb.name, "output": out.name, **plan}, indent=2), encoding="utf-8")
    print(f"✅ Normalised KB: {out}\nReport: {report}")
    return plan


if __name__ == "__main__":
    main()
