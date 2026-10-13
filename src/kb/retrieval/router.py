"""Query routing: classify a question with an LLM and plan the search that fits its kind.

Splitting on word patterns ("and", commas, length) misses questions whose parts are hidden ("the name
of the official who signs off on fire reports" is two facts with no "and"), and gives lists and
processes nothing they need. The router asks an LLM what kind of question it is and returns a plan:

    normal        one search (the default; also every fallback)
    multi_intent  independent questions joined in one sentence -> search each, merge
    comparison    the same aspect for two or more agencies/processes -> one search per side, merge
    multi_hop     a chain: a later step needs an earlier step's answer -> search step by step, the
                  next step written from what the previous one found (Router.next_step)
    conditional   the user's situation decides the rule -> also search a plain rule question
    list          items collected from several places -> wider search, one result per section first
    process       ordered steps of one process -> its section's chunks in document order

The retriever (kb.retrieval.retriever) runs the plan; every plan except `normal` also searches the
original question, so a wrong label costs time but never loses what a plain search would find.
Short questions skip the LLM call, and any failure falls back to `normal`.
"""

import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

QUERY_TYPES = ("normal", "multi_intent", "comparison", "multi_hop", "conditional", "list", "process")
ANSWER_SLOT = "{answer}"


@dataclass
class QueryPlan:
    type: str = "normal"
    # multi_intent / comparison: standalone questions; multi_hop: steps, later ones may hold {answer}
    sub_questions: list[str] = field(default_factory=list)
    rewritten: str | None = None  # conditional: the situation restated as a plain rule question


_PROMPT = """You route searches for a knowledge base of Guyana Ministry of Home Affairs process documents \
(General Register Office, police, prisons, immigration, forensics, fire service, juvenile justice, \
community policing, customs anti-narcotics).

Classify the user's question as exactly one type and plan its search:

- "normal": asks one thing, answerable from one place.
- "multi_intent": two or more independent questions in one sentence. Give each as a standalone question \
in "sub_questions".
- "comparison": compares the same aspect across agencies, processes or options. Give one standalone \
question per side in "sub_questions".
- "multi_hop": a chain, where something the question only describes must be identified first ("the \
office that...", "the official who...", "the system used by the unit that..."), and the question asks \
for a detail of it (its name, number, system, date). Give the steps in order in "sub_questions"; write \
{{answer}} in a later step where the earlier step's answer goes.
- "conditional": the user describes a situation (their own, or a case: "I am...", "my...", "if...", \
"we found...", "when a suspect refuses...") and asks what happens or what applies in it. Restate it as a \
plain question about the rule in "rewritten", keeping the facts that decide the rule (nationality, role, \
age, location, what was found).
- "list": the answer is several items, usually from different places: agencies, problems, partners, \
places, documents. Typical forms: "which agencies...", "list the...", "what are the problems...", \
"where is X used", "who does X work with". A list stays "list" even when it also asks what each item is \
for.
- "process": asks for the ordered steps of one process ("walk me through", "what are the stages", "how \
does X go from A to B").

Rules: at most {n} sub_questions. Each must make sense alone: repeat the subject instead of "it" or \
"they", and keep the user's specific terms. Prefer "normal" when unsure.

Examples:
"How many vehicles does the fire service have?" -> {{"type": "normal"}}
"What does a visa extension cost and who signs prison leave forms?" -> {{"type": "multi_intent", \
"sub_questions": ["What does a visa extension cost?", "Who signs leave forms in the Prison Service?"]}}
"How is training recorded in the police versus the prison service?" -> {{"type": "comparison", \
"sub_questions": ["How is training recorded in the Guyana Police Force?", \
"How is training recorded in the Guyana Prison Service?"]}}
"What database does the agency that issues firearm licences use?" -> {{"type": "multi_hop", \
"sub_questions": ["Which agency issues firearm licences?", "What database does {{answer}} use?"]}}
"I'm a Brazilian miner whose work permit expires next week, what should I do?" -> \
{{"type": "conditional", "rewritten": "How does a Brazilian national renew or extend a mining work permit?"}}
"Which agencies still use paper forms?" -> {{"type": "list"}}
"Walk me through how a fire incident report gets approved." -> {{"type": "process"}}

Return only JSON: {{"type": "...", "sub_questions": [...], "rewritten": "..."}}"""

_NEXT_STEP_PROMPT = """You are answering a question one step at a time from search results.

Original question: {question}
Step just searched: {step}
Next step to search: {next}

Search results for the step just searched:
{context}

From the results, find the answer to the step just searched (a name, office, system, document...). \
Write the next step as one standalone search question with {{answer}} replaced by that answer, keeping \
every other word. If the results do not answer the step, return the next step with {{answer}} replaced \
by your best description from the original question.

Return only JSON: {{"answer": "...", "next": "..."}}"""


def _words(question: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", question))


def _clean(items, limit: int) -> list[str]:
    out: list[str] = []
    for item in items or []:
        if isinstance(item, str) and item.strip():
            text = re.sub(r"\s+", " ", item).strip()[:300]
            if text.lower() not in {s.lower() for s in out}:
                out.append(text)
    return out[:limit]


class Router:
    def __init__(self, model: str, api_key: str | None = None, max_sub_questions: int = 3,
                 min_words: int = 5, timeout: float = 20.0):
        self.model = model
        self.max_sub_questions = max_sub_questions
        self.min_words = min_words  # shorter questions are `normal` without an LLM call
        self._kwargs: dict = {"temperature": 0, "num_retries": 1, "timeout": timeout}
        if api_key:
            self._kwargs["api_key"] = api_key

    def _ask(self, system: str, user: str) -> dict:
        import litellm

        response = litellm.completion(model=self.model, response_format={"type": "json_object"},
                                      messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}], **self._kwargs)
        data = json.loads(response.choices[0].message.content or "{}")
        return data if isinstance(data, dict) else {}

    def classify(self, question: str) -> QueryPlan:
        """The question's type and search plan; `normal` when short, unclear, or the call fails."""
        if _words(question) < self.min_words:
            return QueryPlan()
        try:
            data = self._ask(_PROMPT.format(n=self.max_sub_questions), question)
        except Exception as exc:  # routing is an optimisation: never fail the search over it
            log.warning("Query routing failed, searching the question as is: %s", exc)
            return QueryPlan()
        kind = data.get("type")
        if kind not in QUERY_TYPES:
            return QueryPlan()
        subs = _clean(data.get("sub_questions"), self.max_sub_questions)
        rewritten = data.get("rewritten")
        rewritten = re.sub(r"\s+", " ", rewritten).strip()[:300] if isinstance(rewritten, str) else ""
        # A plan missing what its strategy needs is searched as a normal question, under its own label.
        if kind in ("multi_intent", "comparison", "multi_hop") and len(subs) < 2:
            return QueryPlan(kind)
        if kind == "conditional" and not rewritten:
            return QueryPlan(kind)
        return QueryPlan(kind, subs if kind in ("multi_intent", "comparison", "multi_hop") else [],
                         rewritten if kind == "conditional" else None)

    def next_step(self, question: str, step: str, next_step: str, context: str) -> str:
        """`next_step` written as a standalone question from what the search for `step` found."""
        if ANSWER_SLOT not in next_step:
            return next_step
        fallback = next_step.replace(ANSWER_SLOT, "it")
        if not context:
            return fallback
        try:
            data = self._ask(_NEXT_STEP_PROMPT.format(question=question, step=step, next=next_step,
                                                      context=context[:6000]), "Write the next step.")
        except Exception as exc:
            log.warning("Multi-hop step rewrite failed, searching the step as written: %s", exc)
            return fallback
        written = data.get("next")
        if not isinstance(written, str) or not written.strip() or ANSWER_SLOT in written:
            return fallback
        return re.sub(r"\s+", " ", written).strip()[:300]
