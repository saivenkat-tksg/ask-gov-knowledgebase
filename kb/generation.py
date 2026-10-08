"""Answer generation from retrieved context.

`generate_answer` is what `kb ask` and `kb deepeval` use: one LiteLLM call
(KB_GENERATION_MODEL, default gpt-4o-mini) over the numbered context block, and a fixed
refusal without calling the LLM when retrieval found nothing relevant.

Other ways to plug a model in:

1. Plain messages for any LiteLLM model:

    resp = get_retriever().search(QueryRequest(query="..."))
    answer = litellm.completion(model="gpt-4o-mini", messages=build_messages(resp))

2. A LangChain LCEL chain around any chat model (e.g. `ChatLiteLLM` from
   `langchain-litellm`, `ChatOpenAI`, `ChatAnthropic`):

    chain = build_rag_chain(get_lc_retriever(), llm)
    chain.invoke("when are flu clinics open?")
    # -> {"answer": "... [1]", "sources": [Document, ...]}

In both cases "[n]" markers in the answer map to result n's `citation`.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .lc import format_docs
from .schemas import NO_RELEVANT_CONTEXT_MESSAGE, QueryResponse


class GenerationError(RuntimeError):
    """Chat model unreachable, misconfigured or rejected the request."""

# Each rule answers a failure seen in `kb deepeval`: mixing in another service's document, dropping
# the alternatives that came with the fact, turning "both parties" into "you", computing totals the
# source does not state, and a bare "I don't know" when the source covered part of the question.
DEFAULT_SYSTEM_PROMPT = """\
Answer the user's question using only the numbered sources provided. Cite every claim with its source \
label in square brackets, e.g. [1] or [2][3].

- Each source starts with the document it comes from. Use only sources about the service the user is \
asking about: the rules of one service or programme (for example a cash grant) do not apply to another \
(for example death registration), even when the wording is similar.
- Give the complete answer the sources support: with the direct answer, include the alternatives, \
conditions, next steps and contact details the sources give for it.
- Keep the sources' facts and qualifiers exact: who a requirement applies to (you, both parties, each \
applicant, the deceased), amounts, deadlines, and whether a fee is per copy or in total. Do not simplify them.
- Do not work out figures or conclusions the sources do not state, such as a total for several copies. \
Give the stated figures and say what is not stated.
- If the sources answer only part of the question, give that part, say plainly what they do not cover, \
and point the user to the contact (phone, WhatsApp, email or office) the sources give for confirming it.
- If the sources contain nothing relevant, say you don't have that information. Never answer from \
general knowledge."""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEFAULT_SYSTEM_PROMPT),
        ("human", "Sources:\n\n{context}\n\nQuestion: {question}"),
    ]
)


def build_messages(response: QueryResponse, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Sources:\n\n{response.context or '(no sources found)'}\n\nQuestion: {response.query}",
        },
    ]


def generate_answer(response: QueryResponse, model: str, api_key: str | None = None,
                    temperature: float = 0.0) -> str:
    """Answer `response.query` from its numbered context; refuse without an LLM call when nothing was retrieved."""
    if not response.results:
        return NO_RELEVANT_CONTEXT_MESSAGE
    import litellm

    kwargs = {"api_key": api_key} if api_key else {}
    try:
        out = litellm.completion(model=model, messages=build_messages(response), temperature=temperature,
                                 num_retries=2, **kwargs)
    except Exception as exc:
        raise GenerationError(f"Answer generation failed ({model}): {exc}") from exc
    return (out.choices[0].message.content or "").strip()


def build_rag_chain(
    retriever: BaseRetriever, llm: BaseChatModel | Runnable, prompt: ChatPromptTemplate = RAG_PROMPT
) -> Runnable:
    """question -> {"answer": str, "sources": list[Document]}"""
    answer = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda x: format_docs(x["sources"]) or "(no sources found)"))
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(sources=retriever, question=RunnablePassthrough()).assign(answer=answer)
