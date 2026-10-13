"""The knowledge graph's vocabulary: which entities and relations may be extracted, and between what.

The LLM may only use these names. Anything else is dropped during validation, and the names are
also what makes it safe to put labels and relationship types into Cypher text (Cypher can't take
them as parameters).

Graph layout in Neo4j (every node carries `kb` = the Postgres schema, so one Neo4j database can hold
several knowledge bases; `key` is the unique id, prefixed with that schema):

    (:Document {document_id, doc_key, filename, version, current})
      -[:MENTIONS {chunk_ids, quote}]-> (:Entity:<Type> {type, name, aliases, search_text})
    (:Entity)-[:<RELATION> {source_chunk, document_id, version, current, quote}]->(:Entity)

There are no chunk nodes (migration 0007): chunk text and embeddings stay in Postgres, and the
graph refers to chunks only by id (`chunk_ids`, `source_chunk`).
"""

ENTITY_TYPES: dict[str, str] = {
    "Service": "a government service or process a citizen or agency uses (birth registration, TINT waiver)",
    "Office": "an agency, department, office or unit (General Register Office, Ministry of Home Affairs)",
    "Role": "a job title or person role that performs or approves something (Registrar, ISIS Supervisor)",
    "Requirement": "a document, proof or condition someone must provide or meet (medical certificate, valid ID)",
    "Fee": "an amount of money or 'free of charge' for something (GY$1,000)",
    "Form": "a named form or application (Minor Clerical Application Form, Form A)",
    "Region": "a geographic area or place (Region 6, Georgetown)",
    "Law": "a law, act or regulation (Act 44:01, Marriage Act 45:01)",
    "Deadline": "a time limit or processing time (within 14 days, 2 weeks)",
    "Contact": "a phone number, email, WhatsApp number or website",
}

# relation -> (allowed source types, allowed target types)
RELATION_TYPES: dict[str, tuple[set[str], set[str]]] = {
    "REQUIRES": ({"Service", "Form"}, {"Requirement", "Form", "Fee"}),
    "COSTS": ({"Service", "Requirement", "Form"}, {"Fee"}),
    "HANDLED_BY": ({"Service", "Form"}, {"Office", "Role"}),
    "LOCATED_IN": ({"Office"}, {"Region"}),
    "GOVERNED_BY": ({"Service", "Office", "Requirement"}, {"Law"}),
    "HAS_DEADLINE": ({"Service", "Requirement", "Form"}, {"Deadline"}),
    "USES_FORM": ({"Service"}, {"Form"}),
    "PERFORMED_BY": ({"Service"}, {"Role", "Office"}),
    "APPROVES": ({"Role", "Office"}, {"Service", "Requirement", "Form"}),
    "PART_OF": ({"Office", "Service"}, {"Office", "Service"}),
    "CONTACT_FOR": ({"Contact"}, {"Office", "Service"}),
}
RELATION_DESCRIPTIONS: dict[str, str] = {
    "REQUIRES": "the service or form needs this requirement, form or payment",
    "COSTS": "the service, requirement or form costs this fee",
    "HANDLED_BY": "the service or form is handled or processed by this office or role",
    "LOCATED_IN": "the office is located in this region",
    "GOVERNED_BY": "the service, office or requirement is governed by this law",
    "HAS_DEADLINE": "the service, requirement or form has this time limit or processing time",
    "USES_FORM": "the service uses this form",
    "PERFORMED_BY": "a step of the service is performed by this role or office",
    "APPROVES": "the role or office approves this service, requirement or form",
    "PART_OF": "the office or service is part of a larger office or service",
    "CONTACT_FOR": "the contact detail belongs to this office or service",
}

# Structural labels and relationships written from chunk metadata (no LLM).
STRUCTURE_ENTITY_TYPES = {"Service", "Topic"}
ALL_ENTITY_LABELS = set(ENTITY_TYPES) | STRUCTURE_ENTITY_TYPES

# Bump when the prompt or the vocabulary changes; `kb graph rebuild` then re-extracts everything.
EXTRACTION_VERSION = 1
