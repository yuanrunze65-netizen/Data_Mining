
"""Retry only transport/parse failures using the unchanged course evaluator and prompts."""
import argparse
import importlib.util
import json
from pathlib import Path
from time import sleep

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluator", type=Path, required=True)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--prior-score", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output path")
    evaluator = args.evaluator.resolve()
    spec = importlib.util.spec_from_file_location("course_evaluate", evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    report = json.loads(args.prior_score.read_text(encoding="utf-8"))
    if report["mode"] != "judge_api" or report["subset_evaluation"]:
        p.error("Only retry an existing full judge run")
    unresolved = {r["question_id"] for r in report["results"] if r["context_quality"] == "API_FAILED"}
    if not unresolved:
        p.error("No failed judge requests to retry")
    split, model = report["split"], report["judge_model"]
    workspace = evaluator.parent.parent / "agent_workspace"
    questions = {r["question_id"]: r for r in official.load_array(workspace / split / "questions.json")}
    predictions = official.index_predictions(official.load_array(args.predictions), set(questions))
    ref_path = (workspace / "train/reference_answers.json" if split == "train"
                else evaluator.parent / "test_reference_answers.json")
    refs = {r["question_id"]: r for r in official.load_array(ref_path)}
    system = (evaluator.parent / "evaluator_system.txt").read_text(encoding="utf-8")
    template = (evaluator.parent / "evaluator_user.txt").read_text(encoding="utf-8")
    for item in report["results"]:
        qid = item["question_id"]
        if qid not in unresolved:
            continue
        question = questions[qid]["question"]
        if refs[qid]["question"] != question or not predictions[qid]["answer"].strip():
            raise ValueError("Missing prediction or question/reference mismatch")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": template.format(
            context=predictions[qid]["answer"], question=question, ground_truth=refs[qid]["answer"])}]
        for attempt in range(1, 4):
            report["model_calls"] += 1
            try:
                parsed, raw = official.call_judge(args.judge_url, model, messages)
                item.update(status="SCORED", **official.normalize_judgment(parsed), raw_judge_response=raw)
                item.pop("error", None)
                break
            except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    sleep(attempt)
        print(qid, item["context_quality"], flush=True)
    report["recovered_from"] = args.prior_score.name
    report["retried_question_ids"] = sorted(unresolved)
    report.update(official.summarize(report["results"], len(questions)))
    for cik in report["per_company"]:
        rows = [r for r in report["results"] if r["company_cik"] == cik]
        report["per_company"][cik] = official.summarize(rows, len(rows))
    report["average_support_percentage"] = sum(r["support_percentage"] for r in report["per_company"].values()) / len(report["per_company"])
    official.write(args.output, report)

if __name__ == "__main__":
    main()
