"""
Project Decision Benchmark (PDB) Runner

Tests whether the multi-agent debate system produces better project decisions
than a single agent, using retrospective cases from the project's own history.

Each case reconstructs a real decision point from one of the project's 9 debates.
Both conditions receive identical project state; the evaluator scores each on
a decomposed rubric (6 criteria, binary/ternary scoring).

Rubric criteria:
  A. Context Utilization — per-reference binary (did output cite this prior result?)
  B. Confound Detection — per-confound binary (did output flag this issue?)
  C. History Anti-Repetition — per-failure binary (did output avoid this failed approach?)
  D. Experiment Sequencing — ternary (0=none, 1=sequenced, 2=sequenced + kill criteria)
  E. Actionability — ternary (0=vague, 1=concrete, 2=immediately executable)
  F. Scope Awareness — binary (aware of broader project trajectory?)

Usage:
    python3 research_agent/benchmarks/run_project_benchmark.py              # all cases
    python3 research_agent/benchmarks/run_project_benchmark.py medium       # by difficulty
    python3 research_agent/benchmarks/run_project_benchmark.py --list       # list cases
    python3 research_agent/benchmarks/run_project_benchmark.py --case pdb_001  # single case
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Reuse core utilities from the NRB runner
BENCHMARK_DIR = Path(__file__).parent
PROJECT_CASES_DIR = BENCHMARK_DIR / "project_cases"
RESULTS_DIR = BENCHMARK_DIR / "results"
PROJECT_ROOT = BENCHMARK_DIR.parent.parent

# Import shared utilities from the NRB runner
sys.path.insert(0, str(BENCHMARK_DIR))
from run_benchmark import run_claude, parse_eval_json


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_project_cases(filter_difficulty=None, case_id=None):
    """Load project decision test cases, optionally filtered."""
    cases = []
    for f in sorted(PROJECT_CASES_DIR.glob("pdb_*.json")):
        with open(f) as fh:
            case = json.load(fh)
            if case_id and case["id"] != case_id:
                continue
            if filter_difficulty and case["difficulty"] != filter_difficulty:
                continue
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def format_project_state(case):
    """Format the project_state block for inclusion in prompts."""
    ps = case["project_state"]

    # Prior decisions
    decisions_text = ""
    if ps["prior_decisions"]:
        decisions_text = "\n## Decision History\n"
        for d in ps["prior_decisions"]:
            decisions_text += f"- **{d['date']} — {d['title']}**: {d['outcome']}\n"

    # Metrics
    metrics_text = "\n## Metrics\n"
    for k, v in ps["metrics"].items():
        metrics_text += f"- **{k}**: {v}\n"

    return (
        f"## Project State\n{ps['status_snapshot']}\n"
        f"{decisions_text}"
        f"{metrics_text}"
        f"\n## Version History\n{ps['version_history']}\n"
    )


def build_single_agent_prompt(case):
    """Prompt for the single-agent baseline condition."""
    state = format_project_state(case)
    return (
        "You are a research advisor for an ongoing Minecraft AI project that is "
        "training a VQ-VAE + Diffusion Prior pipeline to generate 3D structures "
        "from text descriptions.\n\n"
        "Given the project state below, answer the research question.\n\n"
        f"{state}\n"
        f"## Question\n{case['question']}\n\n"
        "Propose a specific, sequenced experimental plan. For each step, "
        "specify:\n"
        "1. What you are testing\n"
        "2. What result means success (proceed)\n"
        "3. What result means you should abort this direction\n\n"
        "Reference specific prior results and metrics when justifying your "
        "recommendations. Explain what has already been tried and what "
        "patterns you see in the version history.\n\n"
        "Commit to a concrete plan, not a menu of options."
    )


def build_debate_prompt(case):
    """Prompt for the debate system condition."""
    state = format_project_state(case)
    return (
        "Debate this research question for the Minecraft AI project. "
        "This is a benchmark test — do not search for papers or read external "
        "files. Work only from the project state provided below. "
        "Run the full debate process.\n\n"
        f"{state}\n"
        f"## Question\n{case['question']}\n\n"
        "The project trains a VQ-VAE + Diffusion Prior pipeline to generate "
        "3D voxel structures. The project state above is your complete context."
    )


def build_evaluator_prompt(case, single_output, debate_output):
    """Prompt for the evaluator — decomposed rubric scoring."""
    gt = case["ground_truth"]

    # Format ground truth lists
    confounds = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(gt["confounds_discovered_later"]))
    refs = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(gt["good_references"]))
    failures = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(gt["failed_approaches_to_avoid"]))

    return (
        "You are an impartial evaluator for a Project Decision Benchmark. "
        "Score both outputs against the ground truth using the decomposed "
        "rubric below. Return ONLY a JSON object.\n\n"
        f"## Ground Truth\n\n"
        f"### Confounds Discovered Later\n{confounds}\n\n"
        f"### Good Prior Results to Reference\n{refs}\n\n"
        f"### Failed Approaches to Avoid\n{failures}\n\n"
        f"### Actual Decision Made\n{gt['actual_decision']}\n\n"
        f"### Actual Outcome\n{gt['actual_outcome']}\n\n"

        "## Rubric (score each criterion for each output)\n\n"

        "### A. Context Utilization (per-reference, binary)\n"
        "For each item in 'Good Prior Results to Reference': did the output "
        "reference this specific prior result? Score YES or NO for each.\n"
        "Aggregate: count of YES / total references.\n\n"

        "### B. Confound Detection (per-confound, binary)\n"
        "For each item in 'Confounds Discovered Later': did the output "
        "identify this confound or a closely related concern? Score YES or NO.\n"
        "Aggregate: count of YES / total confounds.\n\n"

        "### C. History Anti-Repetition (per-failure, binary)\n"
        "For each item in 'Failed Approaches to Avoid': did the output "
        "recommend this failed approach again? YES=bad, NO=good.\n"
        "Aggregate: count of NO (avoided) / total failures.\n\n"

        "### D. Experiment Sequencing (ternary)\n"
        "0: No sequencing — single monolithic recommendation\n"
        "1: Sequenced steps but no explicit abort/kill conditions\n"
        "2: Cheapest-first sequence with explicit pass/kill criteria per step\n\n"

        "### E. Actionability (ternary)\n"
        '0: Vague ("try improving the model")\n'
        '1: Concrete ("retrain with rotation augmentation")\n'
        "2: Immediately executable (specific hyperparameters, dataset sizes, "
        "evaluation criteria)\n\n"

        "### F. Scope Awareness (binary)\n"
        "Did the output demonstrate awareness of the project's broader "
        "trajectory (not just the immediate question)? YES or NO.\n\n"

        f"## Single Agent Output\n{single_output[:4000]}\n\n"
        f"## Debate System Output\n{debate_output[:6000]}\n\n"

        "Return your evaluation as ONLY a JSON object with this structure:\n"
        '{\n'
        '  "single_agent": {\n'
        '    "context_utilization": {\n'
        '      "per_reference": {"<reference_summary>": "YES|NO", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "confound_detection": {\n'
        '      "per_confound": {"<confound_summary>": "YES|NO", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "history_anti_repetition": {\n'
        '      "per_failure": {"<failure_summary>": "YES_AVOIDED|NO_REPEATED", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "experiment_sequencing": <0|1|2>,\n'
        '    "actionability": <0|1|2>,\n'
        '    "scope_awareness": "YES|NO",\n'
        '    "reasoning": "<brief justification>"\n'
        '  },\n'
        '  "debate_system": {\n'
        '    "context_utilization": {\n'
        '      "per_reference": {"<reference_summary>": "YES|NO", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "confound_detection": {\n'
        '      "per_confound": {"<confound_summary>": "YES|NO", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "history_anti_repetition": {\n'
        '      "per_failure": {"<failure_summary>": "YES_AVOIDED|NO_REPEATED", ...},\n'
        '      "score": <float 0-1>\n'
        '    },\n'
        '    "experiment_sequencing": <0|1|2>,\n'
        '    "actionability": <0|1|2>,\n'
        '    "scope_awareness": "YES|NO",\n'
        '    "reasoning": "<brief justification>"\n'
        '  },\n'
        '  "debate_advantages": "<which criteria did debate score higher on and why>",\n'
        '  "single_advantages": "<which criteria did single agent score higher on and why>"\n'
        '}'
    )


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------

def run_case(case, run_dir):
    """Run a single project decision case through all conditions and evaluate."""
    case_id = case["id"]
    print(f"\n{'='*60}")
    print(f"  {case_id}: {case['source_debate']}")
    print(f"  Difficulty: {case['difficulty']}")
    print(f"  Q: {case['question'][:80]}...")
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
        timeout_sec=3600,
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
        eval_data = {"parse_error": True, "raw_preview": eval_raw[:500]}

    eval_data["case_id"] = case_id
    eval_data["difficulty"] = case["difficulty"]
    eval_data["source_debate"] = case["source_debate"]

    eval_path = run_dir / f"{case_id}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    return eval_data


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def extract_scores(eval_data, condition):
    """Extract numeric scores from one condition's eval data."""
    cond = eval_data.get(condition, {})
    if not cond or isinstance(cond, str):
        return None

    scores = {}

    # Context utilization
    ctx = cond.get("context_utilization", {})
    scores["context_utilization"] = ctx.get("score", 0) if isinstance(ctx, dict) else 0

    # Confound detection
    conf = cond.get("confound_detection", {})
    scores["confound_detection"] = conf.get("score", 0) if isinstance(conf, dict) else 0

    # History anti-repetition
    hist = cond.get("history_anti_repetition", {})
    scores["history_anti_repetition"] = hist.get("score", 0) if isinstance(hist, dict) else 0

    # Ternary scores
    scores["experiment_sequencing"] = cond.get("experiment_sequencing", 0)
    scores["actionability"] = cond.get("actionability", 0)

    # Binary
    scores["scope_awareness"] = 1 if cond.get("scope_awareness") == "YES" else 0

    return scores


def avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else 0


def compute_summary(results):
    """Compute aggregate metrics across all cases."""
    valid = [r for r in results if "parse_error" not in r and "error" not in r]

    if not valid:
        return {"error": "No valid results to summarize", "total": len(results)}

    criteria = [
        "context_utilization", "confound_detection", "history_anti_repetition",
        "experiment_sequencing", "actionability", "scope_awareness",
    ]

    single_scores = {c: [] for c in criteria}
    debate_scores = {c: [] for c in criteria}

    per_case = []
    for r in valid:
        s = extract_scores(r, "single_agent")
        d = extract_scores(r, "debate_system")
        if s is None or d is None:
            continue

        for c in criteria:
            single_scores[c].append(s[c])
            debate_scores[c].append(d[c])

        per_case.append({
            "id": r["case_id"],
            "difficulty": r["difficulty"],
            "single": s,
            "debate": d,
            "debate_advantages": r.get("debate_advantages", ""),
            "single_advantages": r.get("single_advantages", ""),
        })

    summary = {
        "total_cases": len(results),
        "valid_cases": len(per_case),
        "criteria_comparison": {},
        "per_case": per_case,
    }

    for c in criteria:
        s_avg = avg(single_scores[c])
        d_avg = avg(debate_scores[c])
        summary["criteria_comparison"][c] = {
            "single_avg": s_avg,
            "debate_avg": d_avg,
            "lift": round(d_avg - s_avg, 3),
        }

    # Overall composite (equal-weighted average of all criteria, normalized to 0-1)
    def composite(scores_dict, n):
        if n == 0:
            return 0
        total = 0
        for c in criteria:
            vals = scores_dict[c]
            if c in ("experiment_sequencing", "actionability"):
                # Normalize ternary 0-2 to 0-1
                total += avg(vals) / 2
            else:
                total += avg(vals)
        return round(total / len(criteria), 3)

    n = len(per_case)
    summary["single_composite"] = composite(single_scores, n)
    summary["debate_composite"] = composite(debate_scores, n)
    summary["composite_lift"] = round(
        summary["debate_composite"] - summary["single_composite"], 3
    )

    return summary


def print_summary(summary):
    print(f"\n{'='*60}")
    print("  PROJECT DECISION BENCHMARK — RESULTS")
    print(f"{'='*60}")

    if "error" in summary:
        print(f"  {summary['error']}")
        return

    n = summary["valid_cases"]
    print(f"  Cases evaluated: {n}/{summary['total_cases']}\n")

    print(f"  {'Criterion':<28} {'Single':>8} {'Debate':>8} {'Lift':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")

    for c, v in summary["criteria_comparison"].items():
        label = c.replace("_", " ").title()
        print(f"  {label:<28} {v['single_avg']:>8.3f} {v['debate_avg']:>8.3f} {v['lift']:>+8.3f}")

    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'COMPOSITE':<28} {summary['single_composite']:>8.3f} "
          f"{summary['debate_composite']:>8.3f} {summary['composite_lift']:>+8.3f}")

    print(f"\n  Per-Case Breakdown:")
    for pc in summary["per_case"]:
        print(f"\n    {pc['id']} ({pc['difficulty']}):")
        for c in ["context_utilization", "confound_detection",
                   "history_anti_repetition", "experiment_sequencing",
                   "actionability", "scope_awareness"]:
            s = pc["single"].get(c, "?")
            d = pc["debate"].get(c, "?")
            label = c.replace("_", " ").title()
            print(f"      {label:<26} S={s}  D={d}")
        if pc.get("debate_advantages"):
            print(f"      Debate advantages: {pc['debate_advantages'][:100]}")
        if pc.get("single_advantages"):
            print(f"      Single advantages: {pc['single_advantages'][:100]}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--list" in args:
        cases = load_project_cases()
        print(f"Available project decision cases ({len(cases)}):")
        for c in cases:
            print(f"  {c['id']:>10}  [{c['difficulty']:>6}]  {c['source_debate']}")
        return

    case_id = None
    difficulty = None
    for i, a in enumerate(args):
        if a == "--case" and i + 1 < len(args):
            case_id = args[i + 1]
        elif a in ("easy", "medium", "hard"):
            difficulty = a

    cases = load_project_cases(filter_difficulty=difficulty, case_id=case_id)

    if not cases:
        print("No project decision cases found.")
        print("Add JSON case files to: research_agent/benchmarks/project_cases/")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RESULTS_DIR / f"pdb_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project Decision Benchmark")
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
