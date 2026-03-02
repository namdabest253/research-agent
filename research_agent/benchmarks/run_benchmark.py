"""
Novelty Recovery Benchmark Runner

Tests whether the multi-agent debate system can recover non-obvious
research insights given only a problem statement and known ingredients.

Runs each test case through three conditions:
1. Single agent (one Opus call)
2. Debate system (supervisor + advocates)
3. Evaluator (scores both outputs against the paper's actual insight)

Usage:
    python3 research_agent/benchmarks/run_benchmark.py              # all cases
    python3 research_agent/benchmarks/run_benchmark.py easy          # filter by difficulty
    python3 research_agent/benchmarks/run_benchmark.py --list        # list available cases
    python3 research_agent/benchmarks/run_benchmark.py --case case_001  # run single case
"""

import json
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
CASES_DIR = BENCHMARK_DIR / "cases"
RESULTS_DIR = BENCHMARK_DIR / "results"
PROJECT_ROOT = BENCHMARK_DIR.parent.parent


def load_cases(filter_difficulty=None, case_id=None):
    """Load test cases, optionally filtered."""
    cases = []
    for f in sorted(CASES_DIR.glob("case_*.json")):
        with open(f) as fh:
            case = json.load(fh)
            if case_id and case["id"] != case_id:
                continue
            if filter_difficulty and case["difficulty"] != filter_difficulty:
                continue
            cases.append(case)
    return cases


def build_single_agent_prompt(case):
    """Prompt for the single-agent baseline condition."""
    ingredients = "\n".join(f"- {ing}" for ing in case["known_ingredients"])
    return (
        "You are a research scientist. You are given a problem and a set of known "
        "techniques/observations. Propose a novel solution.\n\n"
        f"## Problem\n{case['problem_statement']}\n\n"
        f"## Known Techniques and Observations\n{ingredients}\n\n"
        "## Your Task\n"
        "Propose a specific, actionable research approach to solve this problem. "
        "Think carefully about what combination of techniques, or what reframing "
        "of the problem, could lead to a breakthrough. Be specific about the "
        "mechanism — don't just name-drop techniques, explain HOW and WHY they "
        "would work together.\n\n"
        "Commit to a single best recommendation. Explain your reasoning step by step."
    )


def build_debate_prompt(case):
    """Prompt for the debate system condition."""
    ingredients = "\n".join(f"- {ing}" for ing in case["known_ingredients"])
    return (
        "Debate this research question. This is a benchmark test — do not search "
        "for papers or use external tools. Work only from the problem statement "
        "and known ingredients below. Run the full debate process.\n\n"
        f"## Problem\n{case['problem_statement']}\n\n"
        f"## Known Techniques and Observations\n{ingredients}\n\n"
        "The goal is to propose a novel, specific solution to this problem."
    )


def build_evaluator_prompt(case, single_output, debate_output):
    """Prompt for the evaluator agent."""
    components = "\n".join(f"- {c}" for c in case["insight_components"])
    return (
        "You are an impartial evaluator for a research novelty benchmark. "
        "Score both outputs against the paper's actual insight.\n\n"
        f"## The Paper's Actual Key Insight\n{case['key_insight']}\n\n"
        f"## Insight Components (check each independently)\n{components}\n\n"
        f"## Single Agent Output\n{single_output[:3000]}\n\n"
        f"## Debate System Output\n{debate_output[:5000]}\n\n"
        "## Scoring Instructions\n\n"
        "### Insight Recovery (0-5) — score each output separately\n"
        "0=No relevant insight, 1=Right area but obvious solution, "
        "2=Right direction but missed core mechanism, "
        "3=Recovered 1-2 insight components, "
        "4=Recovered the key insight in substance, "
        "5=Recovered key insight AND identified new implications\n\n"
        "### Novelty (0-3) — score each output separately\n"
        "0=Only proposed explicitly mentioned techniques, "
        "1=Combined stated techniques non-trivially, "
        "2=Introduced a new technique/framing, "
        "3=Introduced a RELEVANT new technique/framing\n\n"
        "### Debate Value (0-3) — comparing the two outputs\n"
        "0=Substantively identical, 1=Debate added one consideration, "
        "2=Debate changed recommendation meaningfully, "
        "3=Key insight came from cross-examination or Contrarian\n\n"
        "### Component Recovery — for each output\n"
        "For each insight component: RECOVERED / PARTIAL / MISSED\n\n"
        "Return your evaluation as ONLY a JSON object (no other text):\n"
        '{\n'
        '  "single_agent": {\n'
        '    "insight_recovery": <0-5>,\n'
        '    "novelty": <0-3>,\n'
        '    "component_recovery": {"<component>": "RECOVERED|PARTIAL|MISSED", ...},\n'
        '    "reasoning": "<why these scores>"\n'
        '  },\n'
        '  "debate_system": {\n'
        '    "insight_recovery": <0-5>,\n'
        '    "novelty": <0-3>,\n'
        '    "component_recovery": {"<component>": "RECOVERED|PARTIAL|MISSED", ...},\n'
        '    "reasoning": "<why these scores>"\n'
        '  },\n'
        '  "debate_value": <0-3>,\n'
        '  "debate_value_reasoning": "<what the debate added>",\n'
        '  "contrarian_contributed": <true|false>,\n'
        '  "contrarian_contribution": "<what the contrarian added, if anything>"\n'
        '}'
    )


def run_claude(prompt, output_file, use_agent=None, timeout_sec=600):
    """
    Invoke Claude Code CLI non-interactively.

    Args:
        prompt: The prompt text
        output_file: Path to save output
        use_agent: Optional agent name (e.g., 'research-supervisor')
        timeout_sec: Timeout in seconds
    Returns:
        The text output
    """
    cmd = ["claude", "--print", "--output-format", "text"]
    if use_agent:
        cmd.extend(["--agent", use_agent])
    cmd.append(prompt)

    # Strip CLAUDECODE so nested `claude` invocations aren't blocked
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        output = result.stdout or result.stderr or "(no output)"
    except subprocess.TimeoutExpired:
        output = f"(TIMEOUT after {timeout_sec}s)"
    except FileNotFoundError:
        output = "(ERROR: 'claude' CLI not found in PATH)"

    Path(output_file).write_text(output, encoding="utf-8")
    return output


def parse_eval_json(raw_text):
    """Try to extract a JSON object from evaluator output."""
    # Try to find JSON block
    text = raw_text.strip()

    # Look for ```json ... ``` blocks first
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()

    # Find the outermost { ... }
    try:
        brace_start = text.index("{")
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start : i + 1])
    except (ValueError, json.JSONDecodeError):
        pass

    return None


def run_case(case, run_dir):
    """Run a single test case through all conditions and evaluate."""
    case_id = case["id"]
    print(f"\n{'='*60}")
    print(f"  {case_id}: {case['paper']['title']}")
    print(f"  Difficulty: {case['difficulty']} | Domain: {case['domain']}")
    print(f"{'='*60}")

    # --- Condition 1: Single agent ---
    print(f"  [1/3] Running single agent (Opus)...")
    single_output = run_claude(
        build_single_agent_prompt(case),
        run_dir / f"{case_id}_single.md",
    )
    print(f"        Done ({len(single_output)} chars)")

    # --- Condition 2: Debate system ---
    print(f"  [2/3] Running debate system...")
    debate_output = run_claude(
        build_debate_prompt(case),
        run_dir / f"{case_id}_debate.md",
        use_agent="research-supervisor",
        timeout_sec=900,  # debates take longer
    )
    print(f"        Done ({len(debate_output)} chars)")

    # --- Condition 3: Evaluate ---
    print(f"  [3/3] Running evaluator...")
    eval_raw = run_claude(
        build_evaluator_prompt(case, single_output, debate_output),
        run_dir / f"{case_id}_eval_raw.md",
    )

    eval_data = parse_eval_json(eval_raw)
    if eval_data is None:
        print(f"        WARNING: Could not parse evaluator JSON")
        eval_data = {"parse_error": True, "raw_preview": eval_raw[:300]}

    eval_data["case_id"] = case_id
    eval_data["difficulty"] = case["difficulty"]
    eval_data["domain"] = case["domain"]
    eval_data["paper_title"] = case["paper"]["title"]

    eval_path = run_dir / f"{case_id}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    return eval_data


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else 0


def compute_summary(results):
    """Compute aggregate metrics."""
    valid = [r for r in results if "parse_error" not in r and "error" not in r]

    if not valid:
        return {"error": "No valid results to summarize", "total": len(results)}

    single_ins = [r["single_agent"]["insight_recovery"] for r in valid]
    debate_ins = [r["debate_system"]["insight_recovery"] for r in valid]
    debate_vals = [r["debate_value"] for r in valid]
    contrarian_hits = sum(1 for r in valid if r.get("contrarian_contributed"))

    by_diff = {}
    for r in valid:
        d = r["difficulty"]
        by_diff.setdefault(d, {"single": [], "debate": []})
        by_diff[d]["single"].append(r["single_agent"]["insight_recovery"])
        by_diff[d]["debate"].append(r["debate_system"]["insight_recovery"])

    return {
        "total_cases": len(results),
        "valid_cases": len(valid),
        "single_agent": {
            "avg_insight": avg(single_ins),
            "recovery_rate": round(sum(1 for s in single_ins if s >= 3) / len(valid), 2),
        },
        "debate_system": {
            "avg_insight": avg(debate_ins),
            "recovery_rate": round(sum(1 for s in debate_ins if s >= 3) / len(valid), 2),
        },
        "debate_lift": round(avg(debate_ins) - avg(single_ins), 2),
        "avg_debate_value": avg(debate_vals),
        "contrarian_rate": round(contrarian_hits / len(valid), 2),
        "by_difficulty": {
            d: {
                "single": avg(v["single"]),
                "debate": avg(v["debate"]),
                "lift": round(avg(v["debate"]) - avg(v["single"]), 2),
                "n": len(v["single"]),
            }
            for d, v in by_diff.items()
        },
        "per_case": [
            {
                "id": r["case_id"],
                "difficulty": r["difficulty"],
                "single": r["single_agent"]["insight_recovery"],
                "debate": r["debate_system"]["insight_recovery"],
                "debate_value": r["debate_value"],
                "contrarian": r.get("contrarian_contributed", False),
            }
            for r in valid
        ],
    }


def print_summary(summary):
    print(f"\n{'='*60}")
    print("  NOVELTY RECOVERY BENCHMARK — RESULTS")
    print(f"{'='*60}")

    if "error" in summary:
        print(f"  {summary['error']}")
        return

    n = summary["valid_cases"]
    print(f"  Cases evaluated: {n}/{summary['total_cases']}\n")

    sa = summary["single_agent"]
    ds = summary["debate_system"]

    print(f"  {'Metric':<32} {'Single':>8} {'Debate':>8} {'Lift':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Avg Insight Recovery (0-5)':<32} {sa['avg_insight']:>8} {ds['avg_insight']:>8} {summary['debate_lift']:>+8}")
    print(f"  {'Recovery Rate (>=3)':<32} {sa['recovery_rate']:>8} {ds['recovery_rate']:>8}")
    print()
    print(f"  Avg Debate Value:             {summary['avg_debate_value']}")
    print(f"  Contrarian Contribution Rate: {summary['contrarian_rate']}")

    if summary["by_difficulty"]:
        print(f"\n  By Difficulty:")
        for d, v in sorted(summary["by_difficulty"].items()):
            print(f"    {d:>8}: single={v['single']}, debate={v['debate']}, "
                  f"lift={v['lift']:+}, n={v['n']}")

    print(f"\n  Per-Case Breakdown:")
    print(f"  {'ID':<12} {'Diff':<8} {'Single':>7} {'Debate':>7} {'DVal':>5} {'Contrarian':>11}")
    print(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*11}")
    for c in summary["per_case"]:
        ctr = "yes" if c["contrarian"] else "no"
        print(f"  {c['id']:<12} {c['difficulty']:<8} {c['single']:>7} {c['debate']:>7} {c['debate_value']:>5} {ctr:>11}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--list" in args:
        cases = load_cases()
        print(f"Available cases ({len(cases)}):")
        for c in cases:
            print(f"  {c['id']:>12}  [{c['difficulty']:>6}]  {c['paper']['title']}")
        return

    case_id = None
    difficulty = None
    for i, a in enumerate(args):
        if a == "--case" and i + 1 < len(args):
            case_id = args[i + 1]
        elif a in ("easy", "medium", "hard"):
            difficulty = a

    cases = load_cases(filter_difficulty=difficulty, case_id=case_id)

    if not cases:
        print("No test cases found.")
        print("Add JSON case files to: research_agent/benchmarks/cases/")
        print("See research_agent/docs/novelty_benchmark.md for the schema.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Novelty Recovery Benchmark")
    print(f"  Cases:   {len(cases)}")
    print(f"  Output:  {run_dir}")

    results = []
    for case in cases:
        try:
            eval_data = run_case(case, run_dir)
            results.append(eval_data)
        except Exception as e:
            print(f"  ERROR on {case['id']}: {e}")
            results.append({"case_id": case["id"], "error": str(e)})

    summary = compute_summary(results)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)


if __name__ == "__main__":
    main()
