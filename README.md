# knowledgebase-askgov

A retrieval pipeline that runs on your own machine:

```
upload ─► load (pdf/docx/html/md/txt/csv/json, OCR for scanned PDF pages) ─► recursive chunking (LangChain) ─► metadata tagging
       ─► OpenAI embeddings via LiteLLM ─► pgvector (Docker, local)

query  ─► embed ─► pgvector HNSW search + keyword full-text search (hybrid, fused by RRF)
       ─► JSONB metadata filters on both ─► Cohere rerank ─► relevance cut (score ≥ 0.3)
       ─► top-k chunks with citations + a numbered `context` block (ready for an LLM)
```

There's no LLM generation step yet. The pipeline is also available as a LangChain retriever, and [kb/generation.py](kb/generation.py) has a ready-made RAG chain. Adding a model later takes one line.

## Setup (Windows / PowerShell)

```powershell
docker compose up -d                 # pgvector/pgvector:pg17 on localhost:5433 (5433 avoids a local Postgres on 5432)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env               # then set OPENAI_API_KEY and COHERE_API_KEY
kb init                              # creates extension, tables, HNSW + GIN indexes (runs all migrations)
```

### Database migrations (Alembic)

The schema is managed by Alembic migrations in [kb/migrations/versions/](kb/migrations/versions/). Each `KB_DB_SCHEMA` keeps its own `alembic_version` table, so several knowledge bases can share a database. The API and CLI only *check* that the schema is current at startup and never change it; apply migrations as a deploy step:

```powershell
kb migrate --status                  # current vs latest revision
kb migrate                           # upgrade to latest (same as kb init)
kb migrate -1                        # roll back one revision
alembic revision -m "add review_by"  # new migration file; write the SQL in upgrade()/downgrade()
```

Revision `0001` is a baseline written with `IF NOT EXISTS`, so a database created before Alembic was added is adopted by `kb migrate` without changes.

If `COHERE_API_KEY` is empty, results come back in vector-similarity order and `reranked` is `false`.

## Use it

### CLI

```powershell
kb ingest samples --metadata '{\"department\": \"health\", \"year\": 2025, \"tags\": [\"flu\"]}'
kb query "where can I get a free flu shot" --filter '{\"department\": \"health\"}'
kb query "high dose vaccine" --json
kb ingest "MoHA Booklet_V4.pdf" --doc-key moha-booklet --review-by 2027-06-30   # new version
kb list                              # --all includes superseded versions
kb refresh                           # re-check every document's source file/URL (see Knowledge drift)
kb stale                             # documents that may be out of date
kb delete <document-id>
kb serve --reload                    # http://127.0.0.1:8000/docs
```

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | multipart: `files` (one or more), `metadata` (JSON string), `replace` (bool) |
| `GET` | `/documents` | list documents |
| `GET/DELETE` | `/documents/{id}` | get or delete a document along with its chunks |
| `POST` | `/query` | search |
| `GET` | `/health` | DB + pgvector version |

```powershell
curl.exe -F "files=@samples/flu-clinics.md" -F 'metadata={"department":"health","year":2025}' http://127.0.0.1:8000/documents

curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -d '{"query":"flu clinic hours","top_k":3,"filters":{"year":{"$gte":2024}}}'
```

Example response (shortened):

```json
{
  "query": "flu clinic hours",
  "reranked": true,
  "results": [{
    "rank": 1,
    "text": "## Locations and hours\n\nVaccines are offered at ...",
    "vector_score": 0.61, "rerank_score": 0.93,
    "citation": {"label": "[1]", "filename": "flu-clinics.md", "page": null,
                 "chunk_index": 1, "char_start": 312, "char_end": 540,
                 "document_id": "…", "locator": "flu-clinics.md"},
    "metadata": {"department": "health", "year": 2025, "file_type": "markdown", "...": "..."}
  }],
  "context": "[1] flu-clinics.md\n## Locations and hours\n..."
}
```

## Metadata and filters

Every chunk stores a JSONB `metadata` object. It combines three sources:
- the metadata you supply at upload
- metadata found in the file itself, such as a PDF or DOCX title or an HTML `<title>`
- reserved keys the pipeline sets: `document_id`, `filename`, `file_type`, `chunk_index`, `page`, `char_start`, `char_end`, `uploaded_at`

You can't set the reserved keys yourself. You can filter on any key:

```jsonc
{"department": "health"}                          // equality (GIN-indexed containment)
{"year": {"$gte": 2020, "$lt": 2025}}             // numbers, or ISO date strings
{"file_type": {"$in": ["pdf", "docx"]}}
{"tags": {"$contains": "covid"}}                  // array membership
{"author": {"$exists": true}}
{"source.agency": "CDC"}                          // dotted path into nested objects
{"$or": [{"department": "health"}, {"$not": {"draft": true}}]}
```

The supported operators are `$eq $ne $in $nin $contains $gt $gte $lt $lte $exists $and $or $not`. Keys are validated and passed as bind parameters.

Filtered searches set `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) and raise `hnsw.ef_search` to the number of candidates. This way, a selective filter still returns a full candidate set.

## Hybrid search and relevance

Each query runs two searches, both restricted by your metadata filters:
- **Vector search** (pgvector HNSW) finds chunks with similar meaning.
- **Keyword search** (PostgreSQL full-text search on the generated `chunks.content_tsv` column, GIN-indexed) finds chunks that share exact words: phone numbers, fees, form names, acronyms.

**Merging:** the two lists are combined by Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`, so a chunk found by both methods ranks first.

**Ranking and the relevance cut:** Cohere reranks the merged list. Results scoring below `KB_MIN_RERANK_SCORE` (default 0.5) are dropped, and so are results whose cosine similarity is below `KB_MIN_VECTOR_SCORE` (default 0.245): Cohere gives contentless queries such as "hii" high scores against short generic chunks, and the vector floor catches those. If nothing passes, the response has `status: "no_relevant_context"` and empty `results`.

**Response fields:** each result reports `found_by` (`vector`, `keyword` or `both`) and a `keyword_score`.

**Turning hybrid off:** send `"hybrid": false` in the request, or use `kb query ... --no-hybrid`.

**Keyword matching details:**
- A chunk matches if it contains *any* of the question's words after stemming and stopword removal. `ts_rank_cd` favours chunks that match more of the words, close together.
- The ranking is close to BM25, but not exact: it has no IDF weighting.
- For true BM25, switch the database image to ParadeDB (`pg_search`). The Cohere rerank step makes this unnecessary at small scale.

**When hybrid changes results:** with fewer chunks than `candidate_k` (default 40), vector search already passes every chunk to the reranker. Hybrid starts adding recall once the knowledge base is larger than that.

**If Cohere fails or is rate-limited:** results fall back to fused order and the response includes a `warnings` entry. The cut then compares cosine similarity against `KB_MIN_VECTOR_SCORE`, which is less reliable.

## LangChain

LangChain is used for two things:

- **Chunking:** `langchain-text-splitters`' `RecursiveCharacterTextSplitter`, using its Markdown heading separators plus sentence boundaries. Character offsets for citations are computed in [kb/chunking.py](kb/chunking.py) instead of with LangChain's `add_start_index`. That option subtracts the overlap in tokens from a length in characters, so it can return `-1` when chunks are measured in tokens.
- **Composability:** [kb/lc.py](kb/lc.py) wraps the pipeline as a `BaseRetriever`. Each result becomes a `Document` whose metadata holds the chunk's metadata, its scores and a `citation`.

```python
from kb.deps import get_lc_retriever
from kb.generation import build_rag_chain

retriever = get_lc_retriever(top_k=3)
docs = retriever.invoke("flu clinic hours", filters={"department": "health"})

# Later, with any LangChain chat model (e.g. ChatLiteLLM from langchain-litellm):
chain = build_rag_chain(retriever, llm)
chain.invoke("when are flu clinics open?")   # {"question", "sources": [Document], "answer": "... [1]"}
```

Some parts intentionally stay outside LangChain:
- **Storage and search** use the project's own pgvector code instead of `langchain-postgres`. That keeps the schema, the HNSW `ef_search` and `iterative_scan` tuning for filtered searches, the GIN-indexed filters and exact page and character citations.
- **Embeddings** go directly through LiteLLM.
- **Reranking** calls the Cohere SDK directly.

Swapping in LangChain wrappers for these would add dependencies without adding capability.

## Configuration

All settings are in `.env`; see [.env.example](.env.example). The ones you'll most likely change:

| Variable | Default | Notes |
|---|---|---|
| `KB_EMBEDDING_MODEL` | `text-embedding-3-small` | Any LiteLLM embedding model string |
| `KB_EMBEDDING_DIM` | `1536` | Must match the model. The table is created with this size. |
| `KB_EMBEDDING_API_BASE` | — | Point at a LiteLLM proxy or Azure endpoint |
| `KB_RERANK_MODEL` | `rerank-v3.5` | Cohere rerank model |
| `KB_CHUNK_SIZE` / `KB_CHUNK_OVERLAP` | `512` / `64` | Measured in tokens (`cl100k_base`) or characters (`KB_CHUNK_LENGTH_UNIT`) |
| `KB_TOPIC_ROUTING` | `true` | Search only the documents of the service a question names (birth, death, marriage, cash grant) |
| `KB_OCR_MODE` | `auto` | `auto` OCRs only PDF pages without a text layer, `force` OCRs every page, `off` disables OCR |
| `KB_OCR_ENGINE` | `rapidocr` | `rapidocr` (pip only) or `tesseract` (needs the Tesseract binary and `pip install -e ".[tesseract]"`) |
| `KB_OCR_DPI` | `300` | Render resolution for OCR; lower is faster, higher helps with small or faint print |
| `KB_MAX_DOCUMENT_AGE_DAYS` | `365` | A document with no `review_by` date counts as stale after this many days; `0` disables |
| `KB_SOURCE_TIMEOUT_S` | `30` | `kb refresh`: download timeout for source URLs |

If you change the embedding model or dimension later, re-ingest into a fresh database. `kb init` refuses to start when the dimension doesn't match the existing table.

## Behaviour notes

- **Dedup:** a file whose bytes match an existing document returns `status: "duplicate"`.
- **Versions:** every document has a `doc_key` (default: the filename; set `--doc-key` / form field `doc_key` when the filename changes between editions, e.g. `Booklet_V2.pdf` → `Booklet_V3.pdf`). Uploading changed content under an existing `doc_key` adds version N+1 and supersedes the previous one; only the current version is searchable. `valid_from` / `valid_until` (YYYY-MM-DD) also keep a version out of search before or after those dates, and `review_by` records when the content should be checked again. `replace=true` deletes earlier versions instead of keeping them. Deleting the current version makes the previous one current again. `kb list --all` and `GET /documents?include_history=true` show superseded versions; `GET /documents/{id}/versions` lists one document's history.
- **Knowledge drift:** see the next section.
- **PDF pages:** each page is chunked on its own, so every chunk gets an exact `page` for its citation.
- **Topic routing:** single-service documents are tagged with a `topic` at ingest (birth, death, marriage, cash_grant; [kb/topics.py](kb/topics.py)). A question that names a topic searches only those documents, then everything if they hold nothing relevant. Turn off with `KB_TOPIC_ROUTING=false` or `"topic_routing": false` per query.
- **Scanned PDFs (OCR):** a page whose text layer has fewer than `KB_OCR_MIN_CHARS` (20) letters or digits is rendered at `KB_OCR_DPI` and read with OCR, so scanned, mixed and photographed PDFs ingest normally. Chunks from those pages carry `"ocr": true` in their metadata, which you can filter on. RapidOCR runs on the CPU at roughly 3–6 seconds per page; the first OCR call loads the models (about 1 second). Text-layer PDFs are not affected. Use `KB_OCR_MODE=force` for PDFs whose text layer is garbled.
- **Structure-aware splitting:** Markdown, HTML and DOCX headings become `#` markers, and the splitter tries heading boundaries before paragraphs, lines, sentences and words.

## Knowledge drift

Documents describe the rules on the day they were written; when the real rules change, the knowledge base drifts out of date without any error. Four pieces handle that:

| Job | How |
|---|---|
| **Detect** | Every version records its `source`: the file's full path for `kb ingest`, or `--source-url` / form field `source_url` for a web address. `kb refresh` (or `POST /documents/refresh`) re-reads each current document's source. URLs use `ETag` / `Last-Modified`, so unchanged files aren't downloaded again. |
| **Update** | Changed content is ingested as the next version of the same `doc_key`: re-chunked, re-embedded, old version superseded and kept as history, user metadata carried over. Unchanged content only updates `last_checked_at`. `--dry-run` reports changes without ingesting. |
| **Expire** | `--valid-until` keeps a version out of search after that date (unchanged). |
| **Warn** | `kb stale` (or `GET /documents/stale`) lists current documents whose `review_by` date has passed, that are older than `KB_MAX_DOCUMENT_AGE_DAYS` with no review date, that have expired, or whose source has disappeared (`source_status = missing`) or can't be read (`error`). Search adds a `warnings` entry for every cited document in that state. |

A vanished source is only flagged, never deleted automatically: the document stays searchable until someone replaces, expires (`--valid-until`) or deletes it.

Documents ingested before this feature have no source. Running `kb ingest <their folder>` once records it: unchanged files come back as `duplicate` but now have a `source`.

### Running it automatically

[scripts/drift-check.ps1](scripts/drift-check.ps1) runs `kb ingest <inbox>` (optional: picks up files dropped into a folder), `kb refresh` and `kb stale`, and appends the output to `logs\drift-YYYY-MM-DD.log`. It exits with code 1 when something needs a person (a missing or unreadable source, a stale document). Schedule it daily with Task Scheduler:

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\scripts\drift-check.ps1`" -Inbox C:\askgov\inbox"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -TaskName "AskGov drift check" -Action $action -Trigger $trigger
```

`kb refresh` re-embeds the whole document when anything in it changes, so each update costs one embedding run for that document.

## Layout

```
kb/
  loaders.py     file bytes -> text sections (+ page / title metadata)
  ocr.py         OCR for PDF pages without a text layer (RapidOCR or Tesseract)
  migrations/    Alembic migrations for the documents/chunks schema
  chunking.py    LangChain recursive splitter + exact char offsets
  topics.py      topic detection for documents and questions (topic routing)
  embeddings.py  LiteLLM embedding client (batched)
  ingest.py      load -> chunk -> tag -> embed -> store (one transaction)
  documents.py   list / version / delete documents, staleness rules
  drift.py       kb refresh: re-check sources, ingest changed files as new versions
  filters.py     JSON filter -> parameterised SQL over JSONB
  rerank.py      Cohere rerank
  retrieve.py    vector search -> rerank -> cited results + context block
  lc.py          LangChain BaseRetriever adapter (Documents with citations)
  generation.py  build_messages() / build_rag_chain() for a future LLM step
  db.py          run/check migrations, pool, pgvector setup
  api.py / cli.py
tests/           unit tests + an end-to-end pgvector test with fake embeddings
```

## Evaluation

Two evaluations read the same file, [eval/questions.jsonl](eval/questions.jsonl). Each line holds a question, an `expected` label (filename plus a phrase from the right chunk) and a `reference` answer in plain text. Neither uses chunk IDs, so the labels still work after re-ingesting or re-chunking.

```powershell
kb eval eval/questions.jsonl --compare --delay 6.5     # exact label matching: hit@1, hit@k, MRR, refusals (cheap)
pip install -e ".[eval]"
kb deepeval eval/questions.jsonl --delay 6.5 --out eval/deepeval.json   # LLM judge (DeepEval), costs judge calls
```

`kb deepeval` generates an answer for each question from the retrieved chunks (`KB_GENERATION_MODEL`, default `gpt-4o-mini`). A judge model (`--judge-model`, default `gpt-4.1`; `gpt-4o-mini` is cheaper but misreads more sources) then scores both the chunks and the answer:

| Group | Metric | Checks |
|---|---|---|
| retrieval | contextual_precision | relevant chunks rank above irrelevant ones |
| retrieval | contextual_recall | the chunks contain everything the reference needs |
| retrieval | contextual_relevancy | how much of the retrieved text is on topic |
| answer | **correctness** | the answer agrees with the `reference` (G-Eval) |
| answer | **faithfulness** | every claim in the answer is supported by the chunks |
| answer | **answer_relevancy** | the answer addresses the question |

`--metrics retrieval|answer|all` picks the groups; the default is `all`. Each score comes with the judge's reason, and the report also stores each generated answer. To try one answer by hand: `kb ask "price of a birth certificate copy"`. Questions without a `reference` (the off-topic ones) are skipped, because `kb eval` checks refusals. Use `--limit 5` or `--only id1,id2` for a quick check. `--delay 6.5` keeps a Cohere trial key under its 10 calls/minute limit.

## Tests

```powershell
pytest                                                   # unit tests, offline
$env:KB_TEST_DATABASE_URL="postgresql://kb:kb@localhost:5433/kb"; pytest   # + end-to-end on pgvector
```
