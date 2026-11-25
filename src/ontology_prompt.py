# ontology_prompt.py — the ontology-guided extraction step of the proposed method.
#
# Uses only the two customisation points that LightRAG 1.4.x exposes for extraction:
#   1. addon_params["entity_types"]  -> the ontology classes instead of the generic types
#   2. the PROMPTS templates         -> an extra "Domain Ontology" section in the system
#                                       prompt and a domain example instead of the generic
#                                       (fiction / stock-market) examples
# Everything else in the pipeline (parsing, chunking, models, retrieval) stays unchanged.
#
# The ontology contributes classes, relation types and a naming vocabulary only.
# Relations between ontology elements are NOT included (see ontology.py).
from ontology import ENTITY_CLASSES, RELATION_TYPES, Ontology, load_ontology, title_case

SECTION_MARKER = "---Examples---"

# edu-core defines Example as "created in the knowledge graph, not in the ontology", which is
# a modelling note rather than a usable instruction, so the extractor gets this wording instead.
CLASS_HINTS = {"Example": "A worked example in the text that illustrates a concept, theorem or method."}

# Worked example on orthogonality (Lay, chapter 6) — outside the corpus (chapters 1–5)
# and outside the ontology, so it teaches the format without leaking corpus knowledge.
# Written for this project; literal braces are doubled because LightRAG calls .format().
DOMAIN_EXAMPLE = """<Entity_types>
[{entity_types}]

<Input Text>
```
A set of vectors {{u1, ..., up}} in R^n is called an orthogonal set if each pair of distinct vectors from the set is orthogonal, that is, if ui · uj = 0 whenever i ≠ j. If S is an orthogonal set of nonzero vectors, then S is linearly independent and hence is a basis for the subspace spanned by S. An orthogonal basis for a subspace W is a basis for W that is also an orthogonal set. If {{u1, ..., up}} is an orthogonal basis for W, then each y in W has weights cj = (y · uj) / (uj · uj) in the linear combination y = c1 u1 + ... + cp up. In Example 2 the vectors u1 = (3, 1, 1) and u2 = (-1, 2, 1) are shown to be orthogonal because u1 · u2 = 0. The Gram-Schmidt process turns any basis of W into an orthogonal basis of W.
```

<Output>
entity{tuple_delimiter}Orthogonal Set{tuple_delimiter}Concept{tuple_delimiter}An orthogonal set is a set of vectors in R^n in which every pair of distinct vectors has dot product zero.
entity{tuple_delimiter}Dot Product{tuple_delimiter}Concept{tuple_delimiter}The dot product of two vectors is the scalar used to test orthogonality; two vectors are orthogonal when their dot product is zero.
entity{tuple_delimiter}Orthogonal Basis{tuple_delimiter}Concept{tuple_delimiter}An orthogonal basis for a subspace W is a basis for W that is also an orthogonal set.
entity{tuple_delimiter}Basis{tuple_delimiter}Concept{tuple_delimiter}An orthogonal set of nonzero vectors forms a basis for the subspace it spans.
entity{tuple_delimiter}Linear Independence{tuple_delimiter}Concept{tuple_delimiter}Linear independence is the property that is guaranteed for every orthogonal set of nonzero vectors.
entity{tuple_delimiter}Linear Independence of Orthogonal Sets{tuple_delimiter}Theorem{tuple_delimiter}The theorem states that an orthogonal set of nonzero vectors in R^n is linearly independent and is a basis for the subspace it spans.
entity{tuple_delimiter}Weights Relative to an Orthogonal Basis{tuple_delimiter}Theorem{tuple_delimiter}The theorem gives the weights of a vector in an orthogonal basis as cj = (y · uj) / (uj · uj).
entity{tuple_delimiter}Gram-Schmidt Process{tuple_delimiter}Method{tuple_delimiter}The Gram-Schmidt process is a procedure that produces an orthogonal basis from any basis of a subspace.
entity{tuple_delimiter}Checking Orthogonality of Two Vectors{tuple_delimiter}Example{tuple_delimiter}A worked example that shows two specific vectors in R^3 are orthogonal by computing that their dot product is zero.
relation{tuple_delimiter}Orthogonal Set{tuple_delimiter}Dot Product{tuple_delimiter}requires, orthogonality{tuple_delimiter}Orthogonal Set requires Dot Product, because orthogonality is defined by a zero dot product.
relation{tuple_delimiter}Orthogonal Basis{tuple_delimiter}Orthogonal Set{tuple_delimiter}requires, definition{tuple_delimiter}Orthogonal Basis requires Orthogonal Set, because an orthogonal basis is defined as a basis that is an orthogonal set.
relation{tuple_delimiter}Orthogonal Basis{tuple_delimiter}Basis{tuple_delimiter}broader, special case{tuple_delimiter}Orthogonal Basis is a narrower kind of Basis.
relation{tuple_delimiter}Linear Independence of Orthogonal Sets{tuple_delimiter}Orthogonal Set{tuple_delimiter}uses, hypothesis{tuple_delimiter}Linear Independence of Orthogonal Sets uses Orthogonal Set in its hypothesis.
relation{tuple_delimiter}Linear Independence of Orthogonal Sets{tuple_delimiter}Linear Independence{tuple_delimiter}uses, conclusion{tuple_delimiter}Linear Independence of Orthogonal Sets uses Linear Independence in its conclusion.
relation{tuple_delimiter}Weights Relative to an Orthogonal Basis{tuple_delimiter}Orthogonal Basis{tuple_delimiter}uses, coordinates{tuple_delimiter}Weights Relative to an Orthogonal Basis uses Orthogonal Basis to compute the weights of a vector.
relation{tuple_delimiter}Gram-Schmidt Process{tuple_delimiter}Orthogonal Basis{tuple_delimiter}uses, construction{tuple_delimiter}Gram-Schmidt Process uses Orthogonal Basis as the object it constructs.
relation{tuple_delimiter}Checking Orthogonality of Two Vectors{tuple_delimiter}Dot Product{tuple_delimiter}example_of, computation{tuple_delimiter}Checking Orthogonality of Two Vectors is an example of computing a Dot Product to test orthogonality.
{completion_delimiter}

"""


def build_ontology_section(onto: Ontology) -> str:
    """Plain-text 'Domain Ontology' section (not yet escaped for str.format)."""
    lines = [
        "---Domain Ontology---",
        "The input text comes from a university linear algebra textbook. Extract the knowledge "
        "graph of the course according to the following domain ontology.",
        "",
        "Entity types (use exactly these values for `entity_type`):",
    ]
    for cls in ENTITY_CLASSES:
        lines.append(f"- {cls}: {CLASS_HINTS.get(cls) or onto.class_definitions.get(cls, '')}")
    lines += [
        "",
        "Relationship types: the FIRST keyword in `relationship_keywords` must be exactly one of the "
        "following, read as `source_entity <type> target_entity`; further keywords may follow:",
    ]
    for rel, meaning in RELATION_TYPES.items():
        lines.append(f"- {rel}: {meaning}")
    lines += [
        "Write the relationship_description so that it states the relation explicitly, "
        "e.g. `X requires Y, because ...`.",
        "",
        "Naming rules:",
        "1. If an entity is one of the canonical course elements listed below (including any of its "
        "synonyms, plural forms or spelling variants), use EXACTLY the canonical name and its listed type.",
        "2. Course concepts that are not in the list may be extracted too: name them in the singular, "
        "in Title Case, without symbols.",
        "3. Do NOT create entities for variable names or symbols (A, B, T, H, V, x, v1, Matrix A, "
        "Vector U, Basis B). Use the concept they stand for instead (Matrix, Vector, Basis).",
        "4. Do NOT create entities for numbered references (Theorem 8, Example 3, Exercise 12, Figure 2, "
        "Section 1.7) or for study aids. Name a theorem by its established name, or by its content if "
        "it has none (e.g. Determinant of a Triangular Matrix). Extract a worked example only if it "
        "illustrates a course element, and name it by what it shows.",
        "",
        "Canonical course elements (Name [type]; synonyms):",
    ]
    for el in onto.elements:
        syn = [s for s in el.alt_labels + el.hidden_labels if s.lower() != el.pref_label.lower()]
        line = f"- {title_case(el.pref_label)} [{el.cls}]"
        if syn:
            line += "; " + ", ".join(syn)
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def apply_ontology_guidance(prompts: dict = None, onto: Ontology = None) -> dict:
    """Modifies the LightRAG PROMPTS dict in place and returns the addon_params to pass to LightRAG.

    Must be called before the knowledge base is built, and only in the ontology run:
    PROMPTS is a module-level dict, so the change lasts for the whole Python process.
    """
    if prompts is None:
        from lightrag.prompt import PROMPTS as prompts
    onto = onto or load_ontology()

    system = prompts["entity_extraction_system_prompt"]
    if SECTION_MARKER not in system:
        raise RuntimeError(
            f"'{SECTION_MARKER}' not found in LightRAG's extraction prompt. "
            "This module targets lightrag-hku 1.4.x (see requirements.txt).")
    if "---Domain Ontology---" not in system:
        section = _escape(build_ontology_section(onto))
        prompts["entity_extraction_system_prompt"] = system.replace(SECTION_MARKER, section + SECTION_MARKER, 1)
    prompts["entity_extraction_examples"] = [DOMAIN_EXAMPLE]

    return {"entity_types": list(ENTITY_CLASSES)}


def render_system_prompt(prompts: dict, entity_types: list) -> str:
    """Renders the extraction system prompt the same way LightRAG 1.4.16 does (operate.py)."""
    from lightrag.constants import DEFAULT_SUMMARY_LANGUAGE
    tuple_d, done_d = prompts["DEFAULT_TUPLE_DELIMITER"], prompts["DEFAULT_COMPLETION_DELIMITER"]
    examples = "\n".join(prompts["entity_extraction_examples"]).format(
        tuple_delimiter=tuple_d, completion_delimiter=done_d,
        entity_types=", ".join(entity_types), language=DEFAULT_SUMMARY_LANGUAGE)
    return prompts["entity_extraction_system_prompt"].format(
        tuple_delimiter=tuple_d, completion_delimiter=done_d,
        entity_types=",".join(entity_types), examples=examples, language=DEFAULT_SUMMARY_LANGUAGE)
