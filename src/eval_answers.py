# eval_answers.py — answer quality and latency of the knowledge bases.
#
# Three stages, each can be re-run; finished work is skipped (resume after a crash):
#
#   python src/eval_answers.py answer  [--limit 5]   ask every question, cache off, LATENCY_RUNS times
#   python src/eval_answers.py judge   [--limit 5]   claim-based metrics + 1-5 score by a stronger judge model
#   python src/eval_answers.py report                means, per question type, Wilcoxon test, latency
#
#   python src/eval_answers.py answer --graph-only --runs 1  ablation: no text chunks in the context,
#   python src/eval_answers.py judge --graph-only --no-faithfulness   the answer relies on the graph alone
#   python src/eval_answers.py report --graph-only
#
#   python src/eval_answers.py review --n 20                 blind CSV for the manual check of the judge
#   python src/eval_answers.py review --score results/judge_review.csv
#
# Metrics (computed by the judge model, definitions follow RAGAS):
#   factual correctness  answer claims vs. reference claims:
#                        precision = answer claims supported by the reference / all answer claims
#                        recall    = reference claims supported by the answer / all reference claims
#                        F1        = 2PR / (P + R)
#                        Our references are short, so precision also drops when an answer adds
#                        CORRECT details; therefore errors are measured separately:
#   contradiction rate   answer claims contradicted by the reference / all answer claims
#   faithfulness         answer claims supported by the retrieved context / all answer claims
#   judge score          1-5 against the reference answer (rubric below); the judge never sees
#                        which knowledge base produced the answer
#
# Raw answers with the retrieved context are written to eval/runs/ (not committed: the context
# contains textbook text); scores without the context go to results/.
import argparse
import asyncio
import csv
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from config import (EVAL_KBS, EVAL_RUNS_DIR, JUDGE_MAX_ASYNC, JUDGE_MODEL, JUDGE_PRICE_INPUT,
                    JUDGE_PRICE_OUTPUT, JUDGE_RPM, JUDGE_THINKING, LATENCY_RUNS, QUERY_MODE,
                    ENABLE_RERANK, QUESTIONS_FILE, RESULTS_DIR, STORAGE_DIR)

METRICS = ["judge_score", "fc_recall", "fc_contradicted", "faithfulness", "fc_precision", "fc_f1"]
TYPES = ["definition", "relation", "prerequisite", "cross-chapter"]


# ====================================================================== questions / files
def load_questions(path: Path = QUESTIONS_FILE, limit: int | None = None) -> list[dict]:
    qs = json.loads(Path(path).read_text(encoding="utf-8"))["questions"]
    if limit:   # pilot: the first `limit` questions of each type, so every type is represented
        per_type = max(1, math.ceil(limit / len(TYPES)))
        picked = [q for t in TYPES for q in [x for x in qs if x["type"] == t][:per_type]]
        qs = picked[:limit]
    return qs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


SUFFIX = ""      # "_graphonly" in the ablation without text chunks (set from the command line)


def raw_path(kb: str) -> Path:
    return EVAL_RUNS_DIR / f"answers_raw_{kb}{SUFFIX}.jsonl"


def judged_path(kb: str) -> Path:
    return RESULTS_DIR / f"answers_judged_{kb}{SUFFIX}.jsonl"


# ====================================================================== stage 1: answers
def context_text(data: dict) -> str:
    """Retrieved context as plain text (what the answering model saw): entities, relations, chunks."""
    d = data.get("data", data) or {}
    parts = ["Entities:"]
    parts += [f"- {e.get('entity_name', '')} ({e.get('entity_type', '')}): {e.get('description', '')}"
              for e in d.get("entities", [])]
    parts.append("Relationships:")
    parts += [f"- {r.get('src_id', '')} -- {r.get('tgt_id', '')}: {r.get('description', '')}"
              for r in d.get("relationships", [])]
    parts.append("Text chunks:")
    parts += [c.get("content", "") for c in d.get("chunks", [])]
    return "\n".join(parts).replace("<SEP>", " | ")


def context_counts(data: dict) -> dict:
    d = data.get("data", data) or {}
    return {k: len(d.get(k, [])) for k in ("entities", "relationships", "chunks")}


async def open_kb_no_cache(kb_dir: Path):
    from ask_kb import open_kb
    rag = await open_kb(kb_dir)
    # LightRAG checks the flag on the cache storage, so it has to be switched off there too:
    # otherwise a repeated question is answered from the cache in ~0.01 s.
    rag.lightrag.enable_llm_cache = False
    rag.lightrag.llm_response_cache.global_config["enable_llm_cache"] = False
    return rag


async def ask_once(rag, question: str) -> dict:
    from lightrag import QueryParam
    import gemini_funcs as gf
    before = dict(gf.stats)
    t0 = time.perf_counter()
    result = await rag.lightrag.aquery_llm(question, param=QueryParam(mode=QUERY_MODE, enable_rerank=ENABLE_RERANK))
    seconds = time.perf_counter() - t0
    delta = {k: gf.stats[k] - before[k] for k in ("llm", "embed", "retries", "input_tokens", "output_tokens")}
    return {"answer": (result.get("llm_response") or {}).get("content") or "",
            "seconds": round(seconds, 3), "api": delta,
            "context": context_text(result), "context_counts": context_counts(result)}


def drop_text_chunks():
    """Ablation: the retrieved context keeps entities and relations (same token budgets) but no text
    chunks, so the answer depends on the graph alone. LightRAG has no switch for this
    (chunk_top_k = 0 means "no limit"), so the chunk selection step is replaced by an empty one."""
    import lightrag.operate as op

    async def no_chunks(*args, **kwargs):
        return []
    op.process_chunks_unified = no_chunks


async def stage_answer(kbs: list[str], questions: list[dict], runs: int, graph_only: bool = False):
    import gemini_funcs as gf
    if graph_only:
        drop_text_chunks()
    for run in range(1, runs + 1):
        # alternate the order of the knowledge bases between runs, so that neither of them
        # is always measured first (network conditions drift over time)
        order = kbs if run % 2 else list(reversed(kbs))
        for kb in order:
            done = {(r["id"], r["run"]) for r in read_jsonl(raw_path(kb))}
            todo = [q for q in questions if (q["id"], run) not in done]
            if not todo:
                continue
            print(f"\n=== {kb}, run {run}/{runs}: {len(todo)} questions ===")
            rag = await open_kb_no_cache(STORAGE_DIR / kb)
            try:
                for q in todo:          # one at a time: latency must not include waiting for other queries
                    r = await ask_once(rag, q["question"])
                    row = {"id": q["id"], "type": q["type"], "run": run, **r}
                    if run > 1:          # the context is needed only once
                        row.pop("context")
                    append_jsonl(raw_path(kb), row)
                    print(f"  {q['id']}  {r['seconds']:6.1f} s  {len(r['answer']):5d} chars"
                          + (f"  ({r['api']['retries']} retries)" if r["api"]["retries"] else ""))
            finally:
                await rag.finalize_storages()
    print(f"\nAnswering model: {gf.stats['llm']} calls, {gf.stats['input_tokens']:,} input / "
          f"{gf.stats['output_tokens']:,} output tokens, ≈ ${gf.cost_usd():.2f}")


# ====================================================================== stage 2: judge
CLAIMS_PROMPT = """The TEXT answers the QUESTION. Break the TEXT into short, atomic, self-contained factual claims
about linear algebra.
- Each claim states one fact and is understandable without the question and without the other claims:
  replace pronouns and words such as "they" or "it" by what they refer to (use the question for that).
- Keep mathematical conditions exact; formulas may stay in plain or LaTeX notation.
- Ignore greetings, headings, meta-statements ("the context says"), and lists of references or sources.
Return JSON: {"claims": ["...", "..."]}. Return an empty list if the text contains no factual claim.
"""

VERIFY_PROMPT = """For each CLAIM decide how it relates to the PREMISE (the premise was written for the QUESTION).
 1 = supported: the premise states the claim or it follows directly from the premise
 0 = not mentioned: the premise neither states nor contradicts the claim
-1 = contradicted: the premise states something incompatible with the claim
Judge the meaning, not the wording: notation (LaTeX or plain text) and phrasing do not matter.
A claim that lists items (e.g. prerequisites or conditions) is supported if the premise contains those items;
additional items in the premise do not matter.
Judge only by the premise, not by your own knowledge.
Return JSON: {"verdicts": [1, 0 or -1 for each claim, in the same order]}.

QUESTION:
{question}

PREMISE:
{premise}

CLAIMS:
{claims}
"""

SCORE_PROMPT = """You are grading an answer to a question about a linear algebra textbook.
Compare the ANSWER with the REFERENCE answer written by the course author.

Score:
5 - correct and complete: every key point of the reference is present, nothing wrong
4 - correct, with minor omissions
3 - partly correct, or an important point of the reference is missing
2 - mostly wrong or very incomplete
1 - wrong, or does not answer the question

Extra correct details beyond the reference are fine. Mathematical errors lower the score.
Return JSON: {{"score": 1-5, "reason": "one or two sentences"}}.

QUESTION:
{question}

REFERENCE:
{reference}

ANSWER:
{answer}
"""

judge_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}


def judge_cost() -> float:
    out = judge_stats["output_tokens"] + judge_stats["thought_tokens"]
    return (judge_stats["input_tokens"] * JUDGE_PRICE_INPUT + out * JUDGE_PRICE_OUTPUT) / 1e6


_INVALID_ESCAPE = re.compile(r'(?<!\\)\\(?![\\"/])')
_LATEX_COMMAND = re.compile(r'(?<!\\)\\(?=[A-Za-z]{2,})')


def parse_json(text: str) -> dict:
    """JSON from the judge. Claims often contain LaTeX (\\mathbf, \\lambda) that the model does not
    escape, which is invalid JSON; such lone backslashes are escaped before a second attempt."""
    # LaTeX commands such as \\frac or \\beta start with valid JSON escapes (\\f, \\b) and would be
    # silently turned into control characters, so a backslash before 2+ letters is always escaped
    text = _LATEX_COMMAND.sub(r"\\\\", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_INVALID_ESCAPE.sub(r"\\\\", text))


_judge_limiter = None
_judge_sem = None


async def judge_json(prompt: str) -> dict:
    from google.genai import types
    import gemini_funcs as gf
    global _judge_limiter, _judge_sem
    if _judge_limiter is None:
        _judge_limiter, _judge_sem = gf.RateLimiter(JUDGE_RPM), asyncio.Semaphore(JUDGE_MAX_ASYNC)

    async def call():
        await _judge_limiter.wait()
        judge_stats["calls"] += 1
        resp = await gf.client.aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(**JUDGE_THINKING) if JUDGE_THINKING else None,
            ),
        )
        u = resp.usage_metadata
        if u:
            judge_stats["input_tokens"] += u.prompt_token_count or 0
            judge_stats["output_tokens"] += u.candidates_token_count or 0
            judge_stats["thought_tokens"] += u.thoughts_token_count or 0
        return parse_json(resp.text)

    async with _judge_sem:
        for attempt in range(1, 4):      # a malformed JSON answer is rare: simply ask again
            try:
                return await gf._with_retry(call, label="judge")
            except json.JSONDecodeError as e:
                if attempt == 3:
                    raise
                print(f"   ⚠️ [judge] malformed JSON ({e}), asking again")


async def extract_claims(text: str, question: str) -> list[str]:
    if not text.strip():
        return []
    out = await judge_json(f"{CLAIMS_PROMPT}\nQUESTION:\n{question}\n\nTEXT:\n{text}")
    return [c for c in out.get("claims", []) if isinstance(c, str) and c.strip()]


def _verdict(x) -> int:
    try:
        return max(-1, min(1, int(x)))
    except (TypeError, ValueError):
        return 1 if x is True else 0


async def verify(premise: str, claims: list[str], question: str) -> list[int]:
    """1 supported / 0 not mentioned / -1 contradicted, one verdict per claim."""
    if not claims:
        return []
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    prompt = VERIFY_PROMPT.replace("{question}", question).replace("{premise}", premise).replace("{claims}", listing)
    out = await judge_json(prompt)
    v = [_verdict(x) for x in out.get("verdicts", [])]
    if len(v) != len(claims):     # malformed answer: count missing verdicts as unsupported
        v = (v + [0] * len(claims))[:len(claims)]
    return v


def _share(verdicts: list[int], value: int) -> float:
    return sum(v == value for v in verdicts) / len(verdicts) if verdicts else 0.0


def claim_metrics(answer_vs_ref: list[int], ref_vs_answer: list[int], answer_vs_context: list[int]) -> dict:
    """RAGAS-style factual correctness and faithfulness from claim verdicts (1 / 0 / -1).
    All metrics are 0 when the answer has no claims."""
    p, r = _share(answer_vs_ref, 1), _share(ref_vs_answer, 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"fc_precision": round(p, 4), "fc_recall": round(r, 4), "fc_f1": round(f1, 4),
            "fc_contradicted": round(_share(answer_vs_ref, -1), 4),
            "faithfulness": round(_share(answer_vs_context, 1), 4),
            "context_contradicted": round(_share(answer_vs_context, -1), 4),
            "n_answer_claims": len(answer_vs_ref),
            "n_reference_claims": len(ref_vs_answer)}


async def reference_claims(questions: list[dict]) -> dict:
    """Reference answers are split into claims once and reused for every knowledge base."""
    path = EVAL_RUNS_DIR / "reference_claims_v2.json"     # v2: claims extracted with the question as context
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [q for q in questions if cache.get(q["id"], {}).get("reference") != q["reference"]]
    for q, claims in zip(missing, await asyncio.gather(*(extract_claims(q["reference"], q["question"]) for q in missing))):
        cache[q["id"]] = {"reference": q["reference"], "claims": claims}
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: v["claims"] for k, v in cache.items()}


async def judge_one(q: dict, row: dict, ref_claims: list[str], faithfulness: bool = True) -> dict:
    answer = row["answer"]
    ans_claims, score = await asyncio.gather(
        extract_claims(answer, q["question"]),
        judge_json(SCORE_PROMPT.format(question=q["question"], reference=q["reference"], answer=answer)))
    checks = [verify(q["reference"], ans_claims, q["question"]), verify(answer, ref_claims, q["question"])]
    if faithfulness:       # the most expensive check: the whole retrieved context is sent to the judge
        checks.append(verify(row["context"], ans_claims, q["question"]))
    a_ref, r_ans, *rest = await asyncio.gather(*checks)
    a_ctx = rest[0] if rest else [0] * len(ans_claims)
    metrics = claim_metrics(a_ref, r_ans, a_ctx)
    if not faithfulness:
        metrics["faithfulness"] = metrics["context_contradicted"] = None
    s = score.get("score")
    return {"id": q["id"], "type": q["type"], "question": q["question"], "answer": answer,
            "judge_score": int(s) if str(s).isdigit() else None, "judge_reason": score.get("reason", ""),
            **metrics,
            "answer_claims": [{"claim": c, "in_reference": v1, "in_context": v2}
                              for c, v1, v2 in zip(ans_claims, a_ref, a_ctx)],
            "reference_claims": [{"claim": c, "in_answer": v} for c, v in zip(ref_claims, r_ans)]}


async def stage_judge(kbs: list[str], questions: list[dict], force: bool = False, faithfulness: bool = True):
    refs = await reference_claims(questions)
    for kb in kbs:
        if force and judged_path(kb).exists():       # re-judge from scratch (e.g. after a prompt change)
            judged_path(kb).unlink()
        raw = {r["id"]: r for r in read_jsonl(raw_path(kb)) if r["run"] == 1}
        done = {r["id"] for r in read_jsonl(judged_path(kb))}
        todo = [q for q in questions if q["id"] in raw and q["id"] not in done]
        missing = [q["id"] for q in questions if q["id"] not in raw]
        if missing:
            print(f"⚠️ {kb}: no answers yet for {', '.join(missing)} (run the 'answer' stage first)")
        print(f"\n=== judging {kb}: {len(todo)} answers ===")
        for i in range(0, len(todo), JUDGE_MAX_ASYNC):
            batch = todo[i:i + JUDGE_MAX_ASYNC]
            rows = await asyncio.gather(*(judge_one(q, raw[q["id"]], refs[q["id"]], faithfulness) for q in batch))
            for row in rows:
                append_jsonl(judged_path(kb), row)
                print(f"  {row['id']}  score {row['judge_score']}  recall {row['fc_recall']:.2f}  "
                      f"contradicted {row['fc_contradicted']:.2f}  precision {row['fc_precision']:.2f}  "
                      + (f"  faithfulness {row['faithfulness']:.2f}" if row["faithfulness"] is not None else ""))
    print(f"\nJudge {JUDGE_MODEL}: {judge_stats['calls']} calls, {judge_stats['input_tokens']:,} input / "
          f"{judge_stats['output_tokens'] + judge_stats['thought_tokens']:,} output tokens, ≈ ${judge_cost():.2f}")


# ====================================================================== stage 3: statistics
def wilcoxon(x: list[float], y: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test for paired samples (zero differences dropped, average ranks
    for ties). Exact p-value for n <= 25, normal approximation with tie correction otherwise.
    Effect size r = Z / sqrt(n)."""
    d = [b - a for a, b in zip(x, y) if b - a != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "z": 0.0, "p": 1.0, "r": 0.0, "method": "no differences"}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                    # average ranks for tied |d|
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    w_minus = sum(r for r, v in zip(ranks, d) if v < 0)
    mean = n * (n + 1) / 4
    ties = defaultdict(int)
    for r in ranks:
        ties[r] += 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    z = (w_plus - mean) / math.sqrt(var) if var > 0 else 0.0
    if n <= 25:                     # exact distribution of W+ over all 2^n sign assignments
        doubled = [int(round(2 * r)) for r in ranks]
        counts = {0: 1}
        for r in doubled:
            nxt = defaultdict(int)
            for s, c in counts.items():
                nxt[s] += c
                nxt[s + r] += c
            counts = nxt
        total = 2 ** n
        w2 = int(round(2 * w_plus))
        lower = sum(c for s, c in counts.items() if s <= w2) / total
        upper = sum(c for s, c in counts.items() if s >= w2) / total
        p, method = min(1.0, 2 * min(lower, upper)), "exact"
    else:
        p = math.erfc(abs(z) / math.sqrt(2))
        method = "normal approximation"
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": round(z, 3), "p": round(p, 4),
            "r": round(z / math.sqrt(n), 3), "method": method}


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency(kb: str) -> dict:
    """Median over the runs of each question, then median and p90 over questions.
    Runs that needed API retries (429/503) are excluded: their time includes waiting for the API."""
    per_q, excluded, tokens = defaultdict(list), 0, []
    for r in read_jsonl(raw_path(kb)):
        if r["api"]["retries"]:
            excluded += 1
            continue
        per_q[r["id"]].append(r["seconds"])
        tokens.append(r["api"]["input_tokens"])
    med = [statistics.median(v) for v in per_q.values()]
    return {"questions": len(med), "runs_excluded_retries": excluded,
            "median_s": round(statistics.median(med), 2) if med else None,
            "p90_s": round(percentile(med, 0.9), 2) if med else None,
            "mean_input_tokens_per_query": round(statistics.mean(tokens)) if tokens else None,
            "per_question_median_s": {k: round(statistics.median(v), 2) for k, v in per_q.items()}}


def stage_report(kbs: list[str]) -> dict:
    base, onto = kbs
    rows = {kb: {r["id"]: r for r in read_jsonl(judged_path(kb))} for kb in kbs}
    ids = sorted(set(rows[base]) & set(rows[onto]))
    if not ids:
        raise SystemExit("No judged answers for both knowledge bases yet (run 'answer' and 'judge').")
    report = {"knowledge_bases": kbs, "questions": len(ids), "overall": {}, "by_type": {}, "latency": {}}

    def summarize(subset):
        out = {}
        for m in METRICS:
            x = [rows[base][i][m] for i in subset if rows[base][i][m] is not None and rows[onto][i][m] is not None]
            y = [rows[onto][i][m] for i in subset if rows[base][i][m] is not None and rows[onto][i][m] is not None]
            if x:          # a metric can be absent (e.g. faithfulness skipped with --no-faithfulness)
                out[m] = {base: round(statistics.mean(x), 3), onto: round(statistics.mean(y), 3),
                          "wilcoxon": wilcoxon(x, y)}
        return out

    report["overall"] = summarize(ids)
    for t in TYPES:
        sub = [i for i in ids if rows[base][i]["type"] == t]
        if sub:
            report["by_type"][t] = {"n": len(sub), **summarize(sub)}
    for kb in kbs:
        report["latency"][kb] = latency(kb)
    lat_ids = sorted(set(report["latency"][base]["per_question_median_s"]) & set(report["latency"][onto]["per_question_median_s"]))
    report["latency"]["wilcoxon"] = wilcoxon([report["latency"][base]["per_question_median_s"][i] for i in lat_ids],
                                             [report["latency"][onto]["per_question_median_s"][i] for i in lat_ids])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers_report{SUFFIX}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    mode = "graph only, no text chunks" if SUFFIX else "hybrid retrieval"
    print(f"=== Answer quality ({mode}): {base} vs {onto} ({len(ids)} questions) ===\n")
    print(f"| Metric | {base} | {onto} | p (Wilcoxon) | r |")
    print("|---|---|---|---|---|")
    for m, v in report["overall"].items():
        w = v["wilcoxon"]
        print(f"| {m} | {v[base]:.3f} | {v[onto]:.3f} | {w['p']:.4f} | {w['r']:.2f} |")
    print(f"\n| Type | n | judge {base} | judge {onto} | p | recall {base} | recall {onto} | p |")
    print("|---|---|---|---|---|---|---|---|")
    for t, v in report["by_type"].items():
        j, f = v["judge_score"], v["fc_recall"]
        print(f"| {t} | {v['n']} | {j[base]:.2f} | {j[onto]:.2f} | {j['wilcoxon']['p']:.3f} "
              f"| {f[base]:.3f} | {f[onto]:.3f} | {f['wilcoxon']['p']:.3f} |")
    print("\n| Latency | median, s | p90, s | input tokens / query | runs excluded (retries) |")
    print("|---|---|---|---|---|")
    for kb in kbs:
        l = report["latency"][kb]
        print(f"| {kb} | {l['median_s']} | {l['p90_s']} | {l['mean_input_tokens_per_query']} | {l['runs_excluded_retries']} |")
    print(f"\nLatency difference: p = {report['latency']['wilcoxon']['p']}\nFull report: {out}")
    return report


# ====================================================================== manual check of the judge
def write_judge_review(kbs: list[str], n: int, path: Path, seed: int = 42) -> int:
    """Blind sample for the author: answers from both knowledge bases, shuffled, without the judge's score.
    The key (which knowledge base, judge score) is stored separately."""
    items = [(kb, r) for kb in kbs for r in read_jsonl(judged_path(kb))]
    rnd = random.Random(seed)
    by_type = defaultdict(list)
    for it in items:
        by_type[it[1]["type"]].append(it)
    sample, t = [], 0
    pools = {k: rnd.sample(v, len(v)) for k, v in by_type.items()}
    while len(sample) < min(n, len(items)):          # stratified: take types in turn
        pool = pools[TYPES[t % len(TYPES)]] if TYPES[t % len(TYPES)] in pools else []
        if pool:
            sample.append(pool.pop())
        t += 1
    rnd.shuffle(sample)
    questions = {q["id"]: q for q in load_questions()}
    key = {}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "question", "reference", "answer", "your score (1-5)", "comment"])
        for i, (kb, r) in enumerate(sample, 1):
            w.writerow([i, r["question"], questions[r["id"]]["reference"], r["answer"], "", ""])
            key[str(i)] = {"kb": kb, "id": r["id"], "judge_score": r["judge_score"]}
    path.with_suffix(".key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return len(sample)


def score_judge_review(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    header = text.splitlines()[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","     # Excel with a European locale
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    key = json.loads(path.with_suffix(".key.json").read_text(encoding="utf-8"))
    col = next(c for c in rows[0] if c.startswith("your score"))
    pairs = [(int(r[col]), key[r["item"]]["judge_score"]) for r in rows
             if r[col].strip() in {"1", "2", "3", "4", "5"} and key[r["item"]]["judge_score"] is not None]
    if not pairs:
        raise SystemExit(f"No scores (1-5) filled in {path}")
    n = len(pairs)
    res = {"judged": n,
           "exact_agreement": round(sum(a == b for a, b in pairs) / n, 3),
           "agreement_within_1": round(sum(abs(a - b) <= 1 for a, b in pairs) / n, 3),
           "mean_abs_difference": round(sum(abs(a - b) for a, b in pairs) / n, 3),
           "mean_author": round(sum(a for a, _ in pairs) / n, 2),
           "mean_judge": round(sum(b for _, b in pairs) / n, 2),
           "pairs": pairs}
    (RESULTS_DIR / "judge_review_agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ====================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="Answer quality and latency of the knowledge bases")
    p.add_argument("stage", choices=["answer", "judge", "report", "review"])
    p.add_argument("--kbs", nargs=2, default=EVAL_KBS, help="baseline and proposed knowledge base folder names")
    p.add_argument("--limit", type=int, default=None, help="pilot: only N questions (spread over the types)")
    p.add_argument("--runs", type=int, default=LATENCY_RUNS, help="times each question is asked (answer stage)")
    p.add_argument("--graph-only", action="store_true",
                   help="ablation: answer without text chunks (separate result files with suffix _graphonly)")
    p.add_argument("--no-faithfulness", action="store_true",
                   help="judge: skip the faithfulness check (the most expensive one)")
    p.add_argument("--n", type=int, default=20, help="review: sample size")
    p.add_argument("--score", type=Path, help="review: labelled CSV -> agreement between author and judge")
    p.add_argument("--force", action="store_true",
                   help="judge: re-judge all answers from scratch; review: overwrite an existing review CSV")
    args = p.parse_args(argv)
    global SUFFIX
    SUFFIX = "_graphonly" if args.graph_only else ""

    if args.stage == "answer":
        asyncio.run(stage_answer(args.kbs, load_questions(limit=args.limit), args.runs, graph_only=args.graph_only))
    elif args.stage == "judge":
        asyncio.run(stage_judge(args.kbs, load_questions(limit=args.limit), force=args.force,
                                faithfulness=not args.no_faithfulness))
    elif args.stage == "report":
        stage_report(args.kbs)
    elif args.score:
        r = score_judge_review(args.score)
        print(f"Author vs judge on {r['judged']} answers: exact {r['exact_agreement']:.0%}, "
              f"within ±1 {r['agreement_within_1']:.0%}, mean |difference| {r['mean_abs_difference']}")
    else:
        path = RESULTS_DIR / "judge_review.csv"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} already exists (it may contain your scores). Use --force to overwrite it.")
        k = write_judge_review(args.kbs, args.n, path)
        print(f"{k} answers -> {path}\nFill 'your score (1-5)' without looking at the judge, then run:\n"
              f"  python src/eval_answers.py review --score {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
