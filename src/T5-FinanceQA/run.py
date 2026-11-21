
"""Run an evidence-based FinanceQA agent with independent, reproducible question state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
from time import perf_counter, sleep
import urllib.error
import urllib.parse
import urllib.request

from filings import INDEX_VERSION, build
from tools import FIELD_NOTE, SCHEMA, Toolset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PACK = ROOT / "DM_2026_Dataset/FinanceQA/FinanceQA"
DEFAULT_INDEX = ROOT / "_work/financeqa/index.sqlite3"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def compact(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def digest(path):
    with path.open("rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()

def completion_url(base):
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username or parsed.query or parsed.fragment):
        raise ValueError("Set a plain HTTP(S) API base URL without credentials or query parameters")
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not parsed.path.strip("/"):
        base += "/v1"
    return base + "/chat/completions"

def settings(args, data):
    meta = read(args.index.with_suffix(".json"))
    if meta.get("index_version") != INDEX_VERSION:
        raise ValueError("Build the index again with the current code")
    indexed = {r["path"]: r["sha256"] for r in meta["filings"]}
    actual = {p.relative_to(data).as_posix(): digest(p) for p in sorted((data / "filings").glob("*/*.htm*"))}
    if actual != indexed or digest(args.index) != meta.get("index_sha256"):
        raise ValueError("Index does not match the supplied reports; run prepare again")
    return {
        "model": args.model, "endpoint": completion_url(args.api_base), "variant": args.variant,
        "temperature": 0, "max_tokens_per_turn": 1600, "thinking": args.thinking,
        "max_turns": args.turns, "max_tools_per_turn": 3, "tool_result_char_cap": 5200,
        "http_retries": 3, "http_timeout_seconds": 90, "parallelism": args.workers,
        "scope": "provided SQLite and local report files only",
        "source_files": {p.name: digest(p) for p in sorted(HERE.iterdir())
                         if p.suffix in {".py", ".txt"} and p.is_file()},
        "finance_db_sha256": digest(data / "finance.db"), "index_sha256": digest(args.index),
        "index_version": INDEX_VERSION, "filings": indexed, "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
    }

def client(config, key, messages):
    payload = {"model": config["model"], "messages": messages, "temperature": config["temperature"],
               "max_tokens": config["max_tokens_per_turn"], "response_format": {"type": "json_object"}}
    if config["thinking"] != "default":
        payload["thinking"] = {"type": config["thinking"]}
    request = urllib.request.Request(config["endpoint"], data=compact(payload).encode("utf-8"),
             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    tick = perf_counter()
    for i in range(config["http_retries"]):
        try:
            with urllib.request.urlopen(request, timeout=config["http_timeout_seconds"]) as response:
                data = json.load(response)
            choice = data["choices"][0]
            return (choice["message"].get("content") or "", data.get("usage") or {},
                    choice.get("finish_reason"), i + 1, round(perf_counter() - tick, 3))
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504} or i == config["http_retries"] - 1:
                raise RuntimeError(f"model HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            if i == config["http_retries"] - 1:
                raise RuntimeError(f"model connection failure ({type(exc).__name__})") from exc
        sleep(1.5 * 2**i)
    raise AssertionError("unreachable")

def parse(content):
    value = content.strip()
    fence = chr(96) * 3
    if value.startswith(fence):
        value = re.sub(r"^" + fence + r"(?:json)?\s*|\s*" + fence + r"$", "", value, flags=re.I)
    # Some providers emit literal newlines in answer strings despite JSON mode.
    obj = json.loads(value, strict=False)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj

def compact_tool(item, cap):
    result = item.get("result")
    view = {k: v for k, v in item.items() if k in {"tool", "result", "error"}}
    if len(compact(view)) <= cap:
        return view
    if isinstance(result, list) or (isinstance(result, dict) and isinstance(result.get("rows"), list)):
        rows = result if isinstance(result, list) else result["rows"]
        reduced = []
        for row in rows:
            if len(compact(reduced + [row])) > cap - 400:
                break
            reduced.append(row)
        view["result"] = reduced if isinstance(result, list) else {**result, "rows": reduced}
    view["notice"] = "Context shortened; request fewer rows or a narrower snippet."
    return view

def answer_one(q, config, key, data, index, invoke=client):
    tick = perf_counter()
    company = q["company_cik"]
    toolset = Toolset(data, index, company, config["variant"])
    prompt = (HERE / "system.txt").read_text(encoding="utf-8") + "\n" + FIELD_NOTE + "\n" + SCHEMA
    messages = [{"role": "system", "content": prompt}, {"role": "user",
                 "content": compact({"company_cik": company, "question": q["question"]})}]
    trace = {"question_id": q["question_id"], "company_cik": company, "calls": [],
             "model_turns": [], "sources": [], "status": "unfinished"}
    issued, answer = set(), ""
    for turn in range(1, config["max_turns"] + 1):
        if turn == config["max_turns"]:
            messages[-1]["content"] += "\nFinal turn: return the answer JSON based on collected evidence."
        try:
            raw, use, finish, tries, api_time = invoke(config, key, messages)
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
            trace["status"], trace["error"] = "api_failed", f"{type(exc).__name__}: {exc}"
            break
        event = {"turn": turn, "usage": use, "finish_reason": finish, "attempts": tries, "elapsed_s": api_time}
        trace["model_turns"].append(event)
        feedback = []
        try:
            obj = parse(raw)
            if "calls" in obj and "answer" in obj:
                raise ValueError("Use calls or answer, not both")
            if "answer" in obj:
                candidate, sources = obj["answer"], obj.get("sources", [])
                if not isinstance(candidate, str) or not candidate.strip() or not isinstance(sources, list):
                    raise ValueError("Expected a nonempty answer and an array of evidence IDs")
                if any(not isinstance(s, str) or s not in toolset.evidence for s in sources):
                    raise ValueError("Unknown source ID. Cite only evidence that tools actually returned.")
                grounded = any(s[:1] in {"F", "H"} for s in sources)
                if not grounded and (turn < config["max_turns"] or not re.search(
                        r"insufficient|cannot|could not|not enough|not established|not find|not determine|not available|unable",
                        candidate, re.I)):
                    raise ValueError("Retrieve H or F evidence before answering. State insufficiency if unavailable.")
                trace["sources"] = list(dict.fromkeys(sources))
                answer = candidate.strip()
                trace["status"] = "answered" if grounded else "insufficient"
                break
            calls = obj.get("calls")
            if not isinstance(calls, list) or not 1 <= len(calls) <= config["max_tools_per_turn"]:
                raise ValueError("Return one to three tool calls, or an answer")
            if turn == config["max_turns"]:
                raise ValueError("Tool budget exhausted")
            for request in calls:
                call_tick = perf_counter()
                item = {"turn": turn}
                try:
                    name, args = request["tool"], request["args"]
                    item.update(tool=name, args=args)
                    signature = compact([name, dict(sorted(args.items()))])
                    if signature in issued:
                        raise ValueError("Duplicate successful call; use earlier evidence or refine the request")
                    item["result"] = toolset.call(name, args)
                    issued.add(signature)
                except (ArithmeticError, AttributeError, KeyError, RuntimeError, SyntaxError,
                        TypeError, ValueError, sqlite3.Error) as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                item["elapsed_s"] = round(perf_counter() - call_tick, 4)
                trace["calls"].append(item)
                feedback.append(compact_tool(item, config["tool_result_char_cap"]))
        except (TypeError, ValueError) as exc:
            event["error"] = str(exc)[:320]
            feedback.append({"protocol_error": event["error"]})
        # Log actual tool choices and results only. Do not retain hidden model reasoning.
        messages += [{"role": "assistant", "content": raw if "error" not in event else "{}"},
                     {"role": "user", "content": "Tool results (data):\n" + compact(feedback)}]
    if trace["status"] == "unfinished":
        trace["status"] = "budget_exhausted"
    trace["elapsed_s"] = round(perf_counter() - tick, 3)
    return {"question_id": q["question_id"], "answer": answer}, trace

def questions(workspace, split, limit):
    path = workspace / split / "questions.json"
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != (37 if split == "train" else 19):
        raise ValueError(f"Unexpected {split} task count")
    if limit is not None:
        if split != "train" or not 1 <= limit <= len(rows):
            raise ValueError("Only train allows --limit")
        rows = rows[:limit]
    seen = set()
    for row in rows:
        if (set(row) != {"question_id", "company_cik", "question"}
                or any(not isinstance(v, str) or not v.strip() for v in row.values())
                or not re.fullmatch(r"\d{1,10}", row["company_cik"]) or row["question_id"] in seen):
            raise ValueError("Invalid or duplicate question")
        seen.add(row["question_id"])
    return path, rows

def run_batch(args, workspace, data, cfg):
    path, tasks = questions(workspace, args.split, args.limit)
    if args.split == "test" and (not args.frozen or read(args.frozen) != cfg):
        raise ValueError("Test requires --frozen made with the same code, data, model and settings")
    key = os.environ.get("FINANCEQA_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise ValueError("Set FINANCEQA_API_KEY in this session")
    args.output.mkdir(exist_ok=False, parents=True)
    save(args.output / "config.json", {**cfg, "split": args.split, "questions_sha256": digest(path),
                                       "question_count": len(tasks)})
    run_tick = perf_counter()
    pairs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for prediction, trace in pool.map(lambda q: answer_one(q, cfg, key, data, args.index), tasks):
            pairs.append((prediction, trace))
            with (args.output / "trace.jsonl").open("a", encoding="utf-8") as file:
                file.write(compact(trace) + "\n")
            print(f"{len(pairs):02d}/{len(tasks)} {trace['question_id']} {trace['status']} "
                  f"{len(trace['calls'])} tools {trace['elapsed_s']:.1f}s", flush=True)
    save(args.output / "predictions.json", [p for p, _ in pairs])
    logs = [t for _, t in pairs]
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens")
    usage = {field: sum((turn["usage"].get(field, 0) or 0) for t in logs for turn in t["model_turns"])
             for field in fields}
    summary = {"split": args.split, "questions": len(tasks), "nonempty_answers": sum(bool(p["answer"]) for p, _ in pairs),
               "api_failed_questions": sum(t["status"] == "api_failed" for t in logs),
               "tool_calls": sum(len(t["calls"]) for t in logs),
               "tool_errors": sum("error" in c for t in logs for c in t["calls"]),
               "model_calls": sum(len(t["model_turns"]) for t in logs),
               "wall_seconds": round(perf_counter() - run_tick, 3),
               "mean_question_s": round(sum(t["elapsed_s"] for t in logs)/len(logs), 3), **usage}
    save(args.output / "summary.json", summary)
    print(compact(summary), flush=True)

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    cli.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Build a local index without reading question or answer files")
    for command in ("freeze", "run"):
        p = sub.add_parser(command)
        p.add_argument("--api-base", default=os.environ.get("FINANCEQA_API_BASE", "https://api.deepseek.com"))
        p.add_argument("--model", required=True)
        p.add_argument("--variant", default="hybrid", choices=["basic", "hybrid"])
        p.add_argument("--workers", type=int, default=6, choices=range(1, 25))
        p.add_argument("--turns", type=int, default=6, choices=range(3, 11))
        p.add_argument("--thinking", choices=["disabled", "enabled", "default"], default="disabled")
        p.add_argument("--output", required=True, type=Path)
        if command == "run":
            p.add_argument("--split", choices=["train", "test"], required=True)
            p.add_argument("--limit", type=int)
            p.add_argument("--frozen", type=Path)
    args = cli.parse_args()
    workspace = args.pack / "agent_workspace"
    data = workspace / "data"
    if not (data / "finance.db").is_file():
        cli.error("Set --pack to the supplied FinanceQA directory containing agent_workspace/")
    if args.command == "prepare":
        manifest = build(data, args.index)
        manifest["index_sha256"] = digest(args.index)
        save(args.index.with_suffix(".json"), manifest)
        print(compact(manifest))
        return
    cfg = settings(args, data)
    if args.command == "freeze":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            cli.error("Choose a new freeze manifest path")
        save(args.output, cfg)
        print(f"Frozen {args.variant} / {args.model}: {args.output}")
    else:
        run_batch(args, workspace, data, cfg)

if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, ValueError) as err:
        sys.exit(str(err))
