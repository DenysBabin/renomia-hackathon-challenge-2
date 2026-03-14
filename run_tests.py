"""
Test runner for Challenge 2: Document Data Extraction.

Runs all training examples through /solve, calculates per-field scores,
tracks time and token usage, saves results for comparison across runs.

Usage:
    python run_tests.py [--server http://localhost:8080]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import httpx

TRAINING_DIR = Path(__file__).parent / "training"
RESULTS_DIR = Path(__file__).parent / "test_results"

# Field scoring types based on README
ENUM_FIELDS = {
    "state", "assetType", "concludedAs", "contractRegime",
    "actionOnInsurancePeriodTermination",
}
DATE_FIELDS = {"startAt", "endAt", "concludedAt"}
NUMBER_FIELDS = {"installmentNumberPerInsurancePeriod", "insurancePeriodMonths"}
STRING_FIELDS = {"contractNumber", "insurerName", "noticePeriod", "regPlate",
                 "latestEndorsementNumber", "note"}
BOOLEAN_FIELDS = set()  # premium.isCollection handled separately
NESTED_FIELDS = {"premium"}


def score_field(field_name: str, expected, actual) -> tuple[float, str]:
    """Score a single field. Returns (score, reason)."""
    # Null handling
    if expected is None and actual is None:
        return 1.0, "both null"
    if expected is None and actual is not None:
        return 0.0, f"expected null, got {actual!r}"
    if expected is not None and actual is None:
        return 0.0, f"expected {expected!r}, got null"

    # Enum: exact match
    if field_name in ENUM_FIELDS:
        if expected == actual:
            return 1.0, "exact match"
        return 0.0, f"expected {expected!r}, got {actual!r}"

    # Date: exact string match
    if field_name in DATE_FIELDS:
        if expected == actual:
            return 1.0, "exact match"
        return 0.0, f"expected {expected!r}, got {actual!r}"

    # Number: ±10% tolerance
    if field_name in NUMBER_FIELDS:
        if expected == actual:
            return 1.0, "exact match"
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == 0:
                return 0.0 if actual != 0 else 1.0, f"expected {expected}, got {actual}"
            ratio = abs(actual - expected) / abs(expected)
            if ratio <= 0.1:
                return 1.0, f"within 10% ({actual} vs {expected})"
            return 0.0, f"outside 10% ({actual} vs {expected})"
        return 0.0, f"type mismatch: {expected!r} vs {actual!r}"

    # String: fuzzy match
    if field_name in STRING_FIELDS:
        if expected == actual:
            return 1.0, "exact match"
        if isinstance(expected, str) and isinstance(actual, str):
            ratio = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
            return ratio, f"fuzzy {ratio:.2f} ({expected!r} vs {actual!r})"
        return 0.0, f"type mismatch: {expected!r} vs {actual!r}"

    # Boolean: exact match
    if isinstance(expected, bool):
        if expected == actual:
            return 1.0, "exact match"
        return 0.0, f"expected {expected}, got {actual}"

    # Fallback: exact match
    if expected == actual:
        return 1.0, "exact match"
    return 0.0, f"expected {expected!r}, got {actual!r}"


def score_premium(expected: dict, actual: dict) -> list[tuple[str, float, str]]:
    """Score the nested premium object."""
    results = []
    if not isinstance(actual, dict):
        actual = {}

    # currency
    exp_c = expected.get("currency")
    act_c = actual.get("currency")
    s, r = score_field("premium.currency", exp_c, act_c)
    # currency is a string field
    if exp_c is not None and act_c is not None and exp_c != act_c:
        ratio = SequenceMatcher(None, str(exp_c).lower(), str(act_c).lower()).ratio()
        s, r = ratio, f"fuzzy {ratio:.2f} ({exp_c!r} vs {act_c!r})"
    elif exp_c == act_c:
        s, r = 1.0, "exact match"
    results.append(("premium.currency", s, r))

    # isCollection (boolean, but can be null)
    exp_i = expected.get("isCollection")
    act_i = actual.get("isCollection")
    if exp_i is None and act_i is None:
        results.append(("premium.isCollection", 1.0, "both null"))
    elif exp_i is None and act_i is not None:
        results.append(("premium.isCollection", 0.0, f"expected null, got {act_i!r}"))
    elif exp_i is not None and act_i is None:
        results.append(("premium.isCollection", 0.0, f"expected {exp_i!r}, got null"))
    elif exp_i == act_i:
        results.append(("premium.isCollection", 1.0, "exact match"))
    else:
        results.append(("premium.isCollection", 0.0, f"expected {exp_i!r}, got {act_i!r}"))

    return results


def run_example(client: httpx.Client, example_path: Path) -> dict:
    """Run a single training example and return detailed results."""
    with open(example_path) as f:
        data = json.load(f)

    input_data = data["input"]
    expected = data["expected_output"]

    # Measure time
    start_time = time.time()
    response = client.post("/solve", json=input_data, timeout=120)
    elapsed = time.time() - start_time

    if response.status_code != 200:
        return {
            "example": example_path.name,
            "error": f"HTTP {response.status_code}: {response.text}",
            "time_seconds": elapsed,
        }

    actual = response.json()

    # Score each field
    field_scores = []
    for field in ["contractNumber", "insurerName", "state", "assetType",
                  "concludedAs", "contractRegime", "startAt", "endAt",
                  "concludedAt", "installmentNumberPerInsurancePeriod",
                  "insurancePeriodMonths", "actionOnInsurancePeriodTermination",
                  "noticePeriod", "regPlate", "latestEndorsementNumber", "note"]:
        score, reason = score_field(field, expected.get(field), actual.get(field))
        field_scores.append({"field": field, "score": score, "reason": reason})

    # Score premium (nested)
    premium_scores = score_premium(
        expected.get("premium", {}),
        actual.get("premium", {}),
    )
    for field, score, reason in premium_scores:
        field_scores.append({"field": field, "score": score, "reason": reason})

    total_score = sum(f["score"] for f in field_scores) / len(field_scores)

    # Document stats
    docs = input_data.get("documents", [])
    total_chars = sum(len(d.get("ocr_text", "")) for d in docs)

    return {
        "example": example_path.name,
        "total_score": round(total_score, 4),
        "time_seconds": round(elapsed, 2),
        "total_input_chars": total_chars,
        "num_documents": len(docs),
        "field_scores": field_scores,
        "actual_output": actual,
        "expected_output": expected,
    }


def main():
    parser = argparse.ArgumentParser(description="Run training tests")
    parser.add_argument("--server", default="http://localhost:8080",
                        help="Server URL (default: http://localhost:8080)")
    args = parser.parse_args()

    # Find training examples
    examples = sorted(TRAINING_DIR.glob("example_*.json"))
    if not examples:
        print(f"No training examples found in {TRAINING_DIR}")
        sys.exit(1)

    print(f"Found {len(examples)} training examples")
    print(f"Server: {args.server}")
    print("=" * 70)

    client = httpx.Client(base_url=args.server)

    # Get metrics before
    try:
        metrics_before = client.get("/metrics").json()
    except Exception:
        metrics_before = {}

    results = []
    for example_path in examples:
        print(f"\nRunning {example_path.name}...", end=" ", flush=True)
        result = run_example(client, example_path)
        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            score = result["total_score"]
            elapsed = result["time_seconds"]
            chars = result["total_input_chars"]
            marker = "OK" if score >= 0.9 else "WARN" if score >= 0.7 else "FAIL"
            print(f"score={score:.4f} time={elapsed:.1f}s chars={chars} [{marker}]")

            # Print failed fields
            for fs in result["field_scores"]:
                if fs["score"] < 1.0:
                    print(f"  x {fs['field']}: {fs['reason']}")

    # Get metrics after
    try:
        metrics_after = client.get("/metrics").json()
    except Exception:
        metrics_after = {}

    client.close()

    # Token usage for this run
    token_usage = {}
    for key in ["gemini_request_count", "prompt_tokens", "completion_tokens", "total_tokens"]:
        before = metrics_before.get(key, 0)
        after = metrics_after.get(key, 0)
        token_usage[key] = after - before

    # Summary
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_score = sum(r["total_score"] for r in valid_results) / len(valid_results)
        total_time = sum(r["time_seconds"] for r in valid_results)
        total_chars = sum(r["total_input_chars"] for r in valid_results)
    else:
        avg_score = 0
        total_time = 0
        total_chars = 0

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(valid_results)}/{len(results)} examples passed")
    print(f"  Average score:  {avg_score:.4f}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Total chars:    {total_chars}")
    print(f"  Tokens used:    {token_usage.get('total_tokens', '?')}")
    print(f"    prompt:       {token_usage.get('prompt_tokens', '?')}")
    print(f"    completion:   {token_usage.get('completion_tokens', '?')}")
    print(f"    requests:     {token_usage.get('gemini_request_count', '?')}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"run_{timestamp}.json"
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "server": args.server,
        "summary": {
            "examples_total": len(results),
            "examples_passed": len(valid_results),
            "average_score": avg_score,
            "total_time_seconds": total_time,
            "total_input_chars": total_chars,
            "token_usage": token_usage,
        },
        "results": results,
    }
    with open(result_file, "w") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
