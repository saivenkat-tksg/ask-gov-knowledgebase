"""Entity resolution: one graph node per real-world thing.

"GRO", "General Register Office" and "General Registrar Office" must become the same Office node,
or facts about the office end up scattered over three nodes. A name is normalised (case, spacing,
ligatures, punctuation, a leading "the"), then mapped through an alias table; the node key is
schema | type | canonical name, so the type must match too (an Office never merges with a Region).

The alias table below covers names in the current documents. Add entries in a JSON file
({"alias": "canonical name"}) and point KB_GRAPH_ALIASES_FILE at it; changing aliases needs
`kb graph rebuild` to re-key existing nodes.
"""

import json
import re
from pathlib import Path

from kb.graph.extract import _LIGATURES

DEFAULT_ALIASES = {
    "gro": "general register office",
    "general registrar office": "general register office",
    "general registry office": "general register office",
    "moha": "ministry of home affairs",
    "ministry of home affair": "ministry of home affairs",
    "canu": "customs anti-narcotic unit",
    "customs anti narcotics unit": "customs anti-narcotic unit",
    "custom anti narcotics unit": "customs anti-narcotic unit",
    "gfsl": "guyana forensic science laboratory",
    "gpf": "guyana police force",
    "gps": "guyana prison service",
    "gfrs": "guyana fire and rescue service",
    "iss": "immigration support services",
    "national identification card": "national id",
    "national id card": "national id",
    "nid": "national id",
}


def normalise(name: str) -> str:
    text = name.translate(_LIGATURES).lower()
    text = re.sub(r"[\"'`()\[\]{}]", "", text)
    text = re.sub(r"[\s_]+", " ", text).strip(" .,;:-")
    return re.sub(r"^the ", "", text)


class Resolver:
    def __init__(self, aliases_file: str | None = None):
        self.aliases = dict(DEFAULT_ALIASES)
        if aliases_file:
            extra = json.loads(Path(aliases_file).read_text(encoding="utf-8"))
            self.aliases.update({normalise(k): normalise(v) for k, v in extra.items()})

    def canonical(self, name: str) -> str:
        norm = normalise(name)
        return self.aliases.get(norm, norm)

    def key(self, kb: str, entity_type: str, name: str) -> str:
        return f"{kb}|{entity_type}|{self.canonical(name)}"
