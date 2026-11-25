# ontology.py — reads the domain ontology (la_ontology) and exposes what the
# ontology-guided method is allowed to use:
#   * classes      (Concept, Theorem, Method, Example) with their definitions,
#   * relation types (requires, uses, broader, example_of, related_to),
#   * a vocabulary of canonical names (skos:prefLabel) with synonyms
#     (skos:altLabel / skos:hiddenLabel) for name normalisation.
#
# Deliberately NOT exposed: the ontology's own relations between elements
# (edu:requires, edu:uses, skos:broader triples). They are the gold standard
# that extracted relations are later compared against, so giving them to the
# extractor would make that comparison circular.
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdflib import Graph, Namespace, RDF
from rdflib.namespace import SKOS

from config import ONTOLOGY_FILES

EDU = Namespace("http://example.org/edu#")

# Classes the extractor may assign (Example exists in edu-core but is populated
# only by the pipeline, never in the ontology itself).
ENTITY_CLASSES = ["Concept", "Theorem", "Method", "Example"]

# Relation types the extractor may use, with the semantics from edu-core.
# Direction is always source -> target as written in the relation row.
# The illustrations use concepts from outside the corpus and the ontology (orthogonality,
# Lay ch. 6), so no gold-standard relation leaks into the prompt (see tests/test_ontology.py).
RELATION_TYPES = {
    "requires": "the source cannot be understood without the target (the target is a prerequisite "
                "of the source), e.g. Orthogonal Basis requires Orthogonal Set",
    "uses": "a theorem or method (source) uses a concept (target) in its statement or steps, "
            "e.g. Gram-Schmidt Process uses Orthogonal Projection",
    "broader": "the source is a special case or narrower kind of the target, "
               "e.g. Orthonormal Basis broader Orthogonal Basis",
    "example_of": "a worked example (source) illustrates a concept, theorem or method (target)",
    "related_to": "any other meaningful link stated in the text (use only if none of the above fits)",
}


def gold_relations(files: tuple = tuple(str(f) for f in ONTOLOGY_FILES)) -> set:
    """(source prefLabel, relation, target prefLabel) triples stated in the ontology.

    Used ONLY for evaluation and for leakage tests — never passed to the extractor.
    """
    g = Graph()
    for f in files:
        g.parse(f)
    props = {"requires": EDU.requires, "uses": EDU.uses, "broader": SKOS.broader}
    triples = set()
    for name, prop in props.items():
        for s, o in g.subject_objects(prop):
            a, b = _literal(g, s, SKOS.prefLabel), _literal(g, o, SKOS.prefLabel)
            if a and b:
                triples.add((a, name, b))
    return triples


@dataclass
class Element:
    iri: str
    cls: str                      # Concept / Theorem / Method
    pref_label: str
    alt_labels: list = field(default_factory=list)
    hidden_labels: list = field(default_factory=list)
    definition: str = ""
    notation: str = ""
    chapter: int = 0              # order of the module (chapter) its topic belongs to

    @property
    def all_labels(self):
        return [self.pref_label, *self.alt_labels, *self.hidden_labels]


@dataclass
class Ontology:
    class_definitions: dict       # class name -> definition text
    elements: list                # list[Element]

    def label_index(self) -> dict:
        """Lower-cased label (pref/alt/hidden) -> Element, for name normalisation."""
        index = {}
        for el in self.elements:
            for label in el.all_labels:
                index.setdefault(label.strip().lower(), el)
        return index


def _literal(g, s, p):
    for o in g.objects(s, p):
        return str(o)
    return ""


@lru_cache(maxsize=None)
def load_ontology(files: tuple = tuple(str(f) for f in ONTOLOGY_FILES)) -> Ontology:
    g = Graph()
    for f in files:
        g.parse(f)

    class_definitions = {}
    for name in ENTITY_CLASSES:
        class_definitions[name] = _literal(g, EDU[name], SKOS.definition)

    elements = []
    for cls in ["Concept", "Theorem", "Method"]:
        for s in g.subjects(RDF.type, EDU[cls]):
            pref = _literal(g, s, SKOS.prefLabel)
            if not pref:
                continue
            topic = g.value(s, EDU.belongsToTopic)
            module = g.value(topic, EDU.partOf) if topic is not None else None
            order = g.value(module, EDU.order) if module is not None else None
            elements.append(Element(
                chapter=int(order) if order is not None else 0,
                iri=str(s),
                cls=cls,
                pref_label=pref,
                alt_labels=sorted(str(o) for o in g.objects(s, SKOS.altLabel)),
                hidden_labels=sorted(str(o) for o in g.objects(s, SKOS.hiddenLabel)),
                definition=_literal(g, s, SKOS.definition),
                notation=_literal(g, s, EDU.notation),
            ))
    elements.sort(key=lambda e: (e.cls, e.pref_label.lower()))
    return Ontology(class_definitions=class_definitions, elements=elements)


SMALL_WORDS = {"of", "and", "the", "a", "an", "to", "in", "for", "on", "by", "with"}


def title_case(label: str) -> str:
    """LightRAG asks for Title Case names; keep inner capitals (e.g. 'LU Factorization')
    and small words in lower case (e.g. 'Kernel of a Linear Transformation')."""
    out = []
    for i, w in enumerate(label.split()):
        if any(c.isupper() for c in w):
            out.append(w)
        elif i > 0 and w in SMALL_WORDS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)
