"""Command line entry point: `kb init | migrate | ingest | query | list | delete | serve`."""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from uuid import UUID

from . import deps
from .loaders import SUPPORTED_EXTENSIONS
from .schemas import QueryRequest


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS))
        else:
            out.append(path)
    return out


def cmd_init(_: argparse.Namespace) -> None:
    from .config import get_settings
    from .db import init_db

    print(f"Schema ready (pgvector {init_db(get_settings())})")


def cmd_migrate(args: argparse.Namespace) -> None:
    from .config import get_settings
    from .db import migrate, schema_revision

    settings = get_settings()
    if not args.status:
        migrate(settings, args.revision)
    current, head = schema_revision(settings)
    state = "up to date" if current == head else "behind, run `kb migrate`"
    print(f"schema {settings.db_schema}: revision {current or 'none'} (latest {head}) - {state}")


def cmd_ingest(args: argparse.Namespace) -> int:
    metadata = json.loads(args.metadata)
    paths = _files(args.paths)
    if args.doc_key and len(paths) > 1:
        print("error: --doc-key can only be used with a single file", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.WARNING, format="  %(message)s")
    logging.getLogger("kb.ocr").setLevel(logging.INFO)
    ingestor = deps.get_ingestor()
    failed = 0
    for path in paths:
        print(f"reading   {path}", flush=True)
        try:
            r = ingestor.ingest_bytes(path.name, path.read_bytes(), metadata, replace=args.replace,
                                      doc_key=args.doc_key, valid_from=args.valid_from,
                                      valid_until=args.valid_until, review_by=args.review_by)
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"error     {path}  {exc}", file=sys.stderr)
            continue
        version = f"  v{r.version}" if r.version else ""
        superseded = f"  (supersedes {r.superseded_id})" if r.superseded_id else ""
        print(f"{r.status:9} {path}  chunks={r.num_chunks}  id={r.document_id}{version}{superseded}")
    return 1 if failed else 0


def cmd_query(args: argparse.Namespace) -> None:
    req = QueryRequest(
        query=args.text,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        filters=json.loads(args.filter) if args.filter else None,
        hybrid=not args.no_hybrid,
        rerank=not args.no_rerank,
    )
    resp = deps.get_retriever().search(req)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    for warning in resp.warnings:
        print(f"warning: {warning}")
    if resp.status == "no_relevant_context":
        print(f"{resp.message}  (nothing scored >= {resp.min_score_applied})")
        return
    print(f"{len(resp.results)} results  reranked={resp.reranked}  hybrid={resp.hybrid}  "
          f"min_score={resp.min_score_applied}  {resp.took_ms} ms\n")
    for r in resp.results:
        score = f"rerank={r.rerank_score:.3f} " if r.rerank_score is not None else ""
        print(f"{r.citation.label} {r.citation.locator}  ({score}cos={r.vector_score:.3f}, found_by={r.found_by})")
        snippet = r.text.replace("\n", " ")
        print(f"    {snippet[:300]}{'...' if len(snippet) > 300 else ''}\n")


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluate

    # Rerank fallbacks are counted and reported below; don't print one log line per failed call.
    logging.getLogger("kb.retrieve").setLevel(logging.ERROR)
    cases = evaluate.load_cases(args.questions)
    configs = evaluate.CONFIGS if args.compare else [("run", not args.no_hybrid, not args.no_rerank)]
    reports = {}
    for name, hybrid, rerank in configs:
        delay = args.delay if rerank else 0.0
        if delay:
            print(f"{name}: waiting {delay:g}s between searches (~{len(cases) * delay / 60:.0f} min)", file=sys.stderr)
        reports[name] = evaluate.run(deps.get_retriever(), cases, args.top_k, hybrid, rerank, args.min_score, delay)

    cols = ["hit@1", "hit@k", "mrr", "recall", "false_refusal", "correct_refusal", "avg_ms"]
    fmt = lambda v: "-" if v is None else f"{v:g}"  # noqa: E731
    print(f"{len(cases)} questions, top_k={args.top_k}\n")
    print(f"{'config':15}" + "".join(f"{c:>16}" for c in cols))
    for name, (_, summary) in reports.items():
        print(f"{name:15}" + "".join(f"{fmt(summary[c]):>16}" for c in cols))
    for name, (_, summary) in reports.items():
        if summary["rerank_fallbacks"]:
            print(f"\nWARNING {name}: rerank failed for {summary['rerank_fallbacks']}/{len(cases)} questions "
                  "(rate limit, network or missing key), so this row is not a valid rerank result. "
                  "A trial Cohere key allows 10 calls/minute: re-run with --delay 6.5.")
    for name, (results, _) in reports.items():
        lines = evaluate.failures(results)
        if lines:
            print(f"\n{name} failures:")
            print("\n".join(f"  {line}" for line in lines))
    if args.out:
        Path(args.out).write_text(evaluate.to_json(reports), encoding="utf-8")
        print(f"\nfull report: {args.out}")


def cmd_ask(args: argparse.Namespace) -> int:
    from .config import get_settings
    from .embeddings import EmbeddingError
    from .generation import GenerationError, generate_answer

    settings = get_settings()
    req = QueryRequest(query=args.text, top_k=args.top_k,
                       filters=json.loads(args.filter) if args.filter else None)
    try:
        resp = deps.get_retriever().search(req)
        answer = generate_answer(resp, args.model or settings.generation_model, settings.openai_api_key)
    except (EmbeddingError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(answer)
    if resp.results:
        print("\nSources:")
        for r in resp.results:
            print(f"  {r.citation.label} {r.citation.filename}" + (f", p. {r.citation.page}" if r.citation.page else ""))
    return 0


def cmd_deepeval(args: argparse.Namespace) -> int:
    from . import evaluate
    from .config import get_settings
    from .generation import generate_answer

    try:
        import deepeval  # noqa: F401
    except ImportError:
        print('DeepEval is not installed: pip install -e ".[eval]"', file=sys.stderr)
        return 1
    from . import deepeval_eval

    logging.getLogger("kb.retrieve").setLevel(logging.ERROR)
    settings = get_settings()
    deepeval_eval.configure_env(settings.openai_api_key)
    metrics = deepeval_eval.METRIC_SETS[args.metrics]
    thresholds = {name: deepeval_eval.DEFAULT_THRESHOLDS[name] for name in metrics}
    answer_model = args.answer_model or settings.generation_model
    needs_answer = any(name in deepeval_eval.ANSWER_METRICS for name in metrics)

    cases = evaluate.load_cases(args.questions)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only.split(","))]
    if args.limit:
        cases = [c for c in cases if c.reference][: args.limit]
    judged = sum(bool(c.reference) for c in cases)
    done = {}
    if args.resume:
        if not args.out:
            print("--resume needs --out (the report file of the interrupted run)", file=sys.stderr)
            return 1
        if Path(args.out).exists():
            done = deepeval_eval.from_json(Path(args.out).read_text(encoding="utf-8"))
            print(f"Resuming from {args.out}: questions already judged with these metrics are reused",
                  file=sys.stderr)
    print(f"Judging {judged} questions with {args.judge_model} ({len(metrics)} metrics each"
          + (f", answers by {answer_model}" if needs_answer else "")
          + "); this calls the judge model several times per question.", file=sys.stderr)

    rerank = not args.no_rerank
    skipped = len(cases) - judged

    def save_progress(partial):
        # Written after every question, so an interrupted run can continue with --resume.
        if args.out:
            summary = deepeval_eval.summarize(partial, thresholds, skipped, rerank)
            Path(args.out).write_text(deepeval_eval.to_json(partial, {**summary, "complete": False}), encoding="utf-8")

    results, summary = deepeval_eval.run(
        deps.get_retriever(), cases, top_k=args.top_k, hybrid=not args.no_hybrid, rerank=rerank,
        min_score=args.min_score, delay=args.delay, model=args.judge_model, thresholds=thresholds,
        answer_fn=(lambda resp: generate_answer(resp, answer_model, settings.openai_api_key)) if needs_answer else None,
        progress=lambda msg: print(msg, file=sys.stderr), done=done, on_result=save_progress,
    )

    print(f"\n{summary['questions_judged']} questions judged by {args.judge_model}, top_k={args.top_k}"
          + (f", answers by {answer_model}" if needs_answer else "")
          + f" ({summary['skipped_no_reference']} without a reference skipped)\n")
    fmt = lambda v: "-" if v is None else f"{v:g}"  # noqa: E731
    print(f"{'metric':24}{'mean':>8}{'pass rate':>12}{'threshold':>12}{'errors':>8}")
    for group, names in (("retrieval", deepeval_eval.RETRIEVAL_METRICS), ("answer", deepeval_eval.ANSWER_METRICS)):
        names = [n for n in names if n in thresholds]
        if names:
            print(f"  {group}")
        for name in names:
            s = summary[name]
            print(f"{name:24}{fmt(s['mean']):>8}{fmt(s['pass_rate']):>12}{thresholds[name]:>12g}{s['errors']:>8}")
    if summary["refused"]:
        print(f"\n{summary['refused']} answerable questions got 'no relevant information' (nothing passed the "
              "relevance cut); they score 0 on retrieval and correctness.")
    if summary["rerank_fallbacks"]:
        print(f"\nWARNING: rerank failed for {summary['rerank_fallbacks']} questions, so their chunks are in "
              "retrieval order. A trial Cohere key allows 10 calls/minute: re-run with --delay 6.5.")
    for name in ("correctness", "faithfulness", "contextual_recall", "contextual_precision"):
        if name not in thresholds:
            continue
        worst = [r for r in deepeval_eval.lowest(results, name) if not r.scores[name].passed]
        if worst:
            print(f"\nlowest {name}:")
            for r in worst:
                reason = (r.scores[name].reason or "").replace("\n", " ")
                print(f"  {r.scores[name].score:<5g} {r.id}: {r.question!r}")
                if r.answer and name in deepeval_eval.ANSWER_METRICS:
                    print(f"        answer: {r.answer.replace(chr(10), ' ')[:200]}")
                print(f"        judge:  {reason[:220]}")
    failed = [r for r in results if r.status == deepeval_eval.PIPELINE_ERROR]
    if failed:
        print(f"\nWARNING: search or answer generation failed for {len(failed)} questions even after retries "
              f"(network or API down?): {', '.join(r.id for r in failed[:10])}. They are excluded from the scores.")
        if args.out:
            print(f"Re-run the same command with --resume to retry only those: "
                  f"kb deepeval {args.questions} --metrics {args.metrics} --resume --out {args.out}")
    errors = [(r.id, n, s.error) for r in results if r.status != deepeval_eval.PIPELINE_ERROR
              for n, s in r.scores.items() if s.error]
    if errors:
        print("\njudge errors:")
        for case_id, name, err in errors[:10]:
            print(f"  {case_id} {name}: {err[:160]}")
    if args.out:
        Path(args.out).write_text(deepeval_eval.to_json(results, {**summary, "complete": not failed}),
                                  encoding="utf-8")
        print(f"\nfull report (scores, answers and judge reasons per question): {args.out}")
    return 1 if failed else 0


def cmd_list(args: argparse.Namespace) -> None:
    from .documents import list_documents

    for d in list_documents(deps.get_pool(), include_history=args.all):
        state = "" if d.is_current else "  [superseded]"
        dates = "".join(f"  {name}={value}" for name, value in
                        (("valid_from", d.valid_from), ("valid_until", d.valid_until), ("review_by", d.review_by))
                        if value)
        print(f"{d.id}  {d.file_type:8} chunks={d.num_chunks:<5} v{d.version} {d.filename}{state}{dates}  "
              f"{json.dumps(d.metadata)}")


def cmd_delete(args: argparse.Namespace) -> int:
    from .documents import delete_document

    if delete_document(deps.get_pool(), UUID(args.document_id)):
        print("deleted")
        return 0
    print("not found", file=sys.stderr)
    return 1


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("kb.api:app", host=args.host, port=args.port, reload=args.reload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb", description="AskGov knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create pgvector extension, tables and indexes (applies all migrations)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("migrate", help="Apply database migrations (Alembic) in KB_DB_SCHEMA")
    p.add_argument("revision", nargs="?", default="head", help="target: head (default), a revision id, -1, or base")
    p.add_argument("--status", action="store_true", help="only show the current and latest revision")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("ingest", help="Ingest files or folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--metadata", default="{}", help='JSON object, e.g. \'{"department":"health"}\'')
    p.add_argument("--replace", action="store_true",
                   help="Delete all earlier versions instead of keeping them as history")
    p.add_argument("--doc-key", help="Stable identity across versions (default: the filename); single file only")
    p.add_argument("--valid-from", type=date.fromisoformat, help="YYYY-MM-DD; not searchable before this date")
    p.add_argument("--valid-until", type=date.fromisoformat, help="YYYY-MM-DD; not searchable after this date")
    p.add_argument("--review-by", type=date.fromisoformat, help="YYYY-MM-DD; date to review the content")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="Search the knowledge base")
    p.add_argument("text")
    p.add_argument("--filter", help='JSON metadata filter, e.g. \'{"year": {"$gte": 2023}}\'')
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--candidate-k", type=int, default=40)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-hybrid", action="store_true", help="Vector search only (skip keyword search)")
    p.add_argument("--json", action="store_true", help="Print the full JSON response")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("eval", help="Score retrieval against a labelled questions file (JSONL)")
    p.add_argument("questions", help="e.g. eval/questions.jsonl")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-score", type=float, help="Override the relevance cut")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--compare", action="store_true", help="Run vector, hybrid and hybrid+rerank side by side")
    p.add_argument("--out", help="Write the per-question JSON report here")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to wait between reranked searches (6.5 for a 10 calls/minute Cohere trial key)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("ask", help="Answer a question from the knowledge base with an LLM, with sources")
    p.add_argument("text")
    p.add_argument("--filter", help='JSON metadata filter, e.g. \'{"department":"health"}\'')
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--model", help="Chat model (default KB_GENERATION_MODEL)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("deepeval", help="LLM-judged retrieval evaluation (DeepEval) using `reference` answers")
    p.add_argument("questions", help="e.g. eval/questions.jsonl")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-score", type=float, help="Override the relevance cut")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds between searches (6.5 for a 10 calls/minute Cohere trial key)")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model used as the judge")
    p.add_argument("--metrics", choices=["all", "retrieval", "answer"], default="all",
                   help="retrieval: contextual precision/recall/relevancy; answer: correctness, faithfulness, "
                        "answer relevancy (generates an answer per question); all: both (default)")
    p.add_argument("--answer-model", help="Model that writes the answers (default KB_GENERATION_MODEL)")
    p.add_argument("--limit", type=int, help="Judge only the first N questions (quick, cheap check)")
    p.add_argument("--only", help="Comma-separated question ids, e.g. death-03,birth-05")
    p.add_argument("--out", help="Write per-question scores and judge reasons here (JSON, saved after every question)")
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run: reuse questions already judged in --out, retry failed ones")
    p.set_defaults(func=cmd_deepeval)

    p = sub.add_parser("list", help="List documents (current versions)")
    p.add_argument("--all", action="store_true", help="Also list superseded versions")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("delete", help="Delete one document version and its chunks (the previous version becomes current)")
    p.add_argument("document_id")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("serve", help="Run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    finally:
        deps.close()


if __name__ == "__main__":
    sys.exit(main())
