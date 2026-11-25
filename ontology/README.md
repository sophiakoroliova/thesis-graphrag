# Domain ontology

Copied from [la_ontology](https://github.com/sophiakoroliova/la_ontology) (commit `006f1f8`, 2026-11-05):

- `edu-core.ttl` — course-independent core: classes (Concept, Theorem, Method, Example, ...) and relations (requires, uses, ...)
- `la-concepts.ttl` — 111 linear algebra elements with labels, definitions, prerequisites
- `la-shapes.ttl` — SHACL shapes for validating the graph

How the method uses it (`src/ontology.py`, `src/ontology_prompt.py`):

| Part of the ontology | Given to the extractor? | Used for |
|---|---|---|
| Classes and their definitions | yes | entity types in the extraction prompt |
| Relation types | yes | allowed relationship keywords |
| `skos:prefLabel` / `altLabel` / `hiddenLabel` | yes | naming vocabulary, name normalisation |
| Relations between elements (`edu:requires`, `edu:uses`, `skos:broader`) | **no** | gold standard for evaluation only |
| SHACL shapes | no | validation of the built graph |
