# Project Decision Benchmark (PDB) — Implementation Summary

**Date**: 2026-03-03
**Status**: Implementation Complete, Smoke Test Running

## What Was Built

### 1. Five Project Decision Cases (JSON)

**Location**: `research_agent/benchmarks/project_cases/`

Each case reconstructs a real decision point from the project's debate history:

- **pdb_001** — v7 approach (2026-02-25): What training config after v6a blobs?
  - Project state: 6 prior continuous diffusion failures, v6a with weak discriminator
  - Confounds to detect: No VQ-VAE structural diagnostic run yet
  - Good references: Adversarial failures (v2, v5, v6a x3), dataset size constraint (3,500→14,000 via rotation)

- **pdb_002** — Scope reduction (2026-02-26): Next step after 7 failures + quantization mismatch refuted?
  - Project state: Quantization bypass = 95.6% voxel match (mismatch hypothesis wrong)
  - Confounds to detect: Broken evaluation criteria, automated clustering ≠ visual homogeneity, oversized model
  - Ground truth outcome: Scope reduction test returned invalid results (0/20 real AND 0/20 generated)

- **pdb_003** — Retest before pivot (2026-02-27): Ambiguous scope-reduction result (criteria broken). What next?
  - Project state: 7 failed versions, test methodology failure, 0/20 on both real and generated
  - Confounds to detect: Noisy tag labels, model too large, lack of validated criteria
  - Ground truth: Manual curation + criteria validation needed

- **pdb_004** — Final clean test (2026-03-01): House-filtered v7 produced blobs + overfitting. Pipeline broken or confounded?
  - Project state: 8th continuous attempt, overfitting (gap=0.14), amorphous blobs again
  - Confounds to detect: Noisy CSV tags, 3.5M params on small dataset, wrong checkpoint
  - Pattern: Project's 4th confounded experiment in a row

- **pdb_005** — Meta-debate (2026-03-03): Highest-ROI debate system improvement after benchmark showed no insight lift?
  - Project state: NRB shows debate=single on insight (3-4/5), but debate better on experimental design
  - Confounds to detect: Unmeasured 4 optimizations applied simultaneously, holistic scoring too coarse
  - Key insight: 1-round vs 2-round comparison is cheapest/highest-info experiment

### 2. Runner Script

**File**: `research_agent/benchmarks/run_project_benchmark.py`

Core features:

- **Case loading**: `load_project_cases()` with difficulty/case_id filters
- **Prompt builders**:
  - `build_single_agent_prompt()` — Full project state dumped into one Opus call
  - `build_debate_prompt()` — Same state provided to supervisor agent
  - `build_evaluator_prompt()` — Decomposed rubric with ground truth
- **Conditions**:
  - Single agent: Opus baseline (600s timeout)
  - Debate: Research-supervisor agent (3600s timeout)
  - Evaluator: Claude evaluates both on 6 rubric criteria
- **Decomposed rubric scoring**:
  - A. Context Utilization (binary per-reference)
  - B. Confound Detection (binary per-confound)
  - C. History Anti-Repetition (binary per-failure)
  - D. Experiment Sequencing (ternary: 0=none, 1=sequenced, 2=+kill criteria)
  - E. Actionability (ternary: 0=vague, 1=concrete, 2=executable)
  - F. Scope Awareness (binary)
- **Metrics**: Per-criterion averages, composite lift, per-case breakdown
- **Output**: JSON eval files + summary.json with per-criterion comparison

Usage:
```bash
python3 research_agent/benchmarks/run_project_benchmark.py --case pdb_001   # single case
python3 research_agent/benchmarks/run_project_benchmark.py                  # all 5 cases
python3 research_agent/benchmarks/run_project_benchmark.py hard             # by difficulty
python3 research_agent/benchmarks/run_project_benchmark.py --list           # list cases
```

### 3. Design Documentation

**File**: `research_agent/docs/project_decision_benchmark.md`

Covers:
- Purpose (why this benchmark matters)
- Case design schema and inventory
- Decomposed rubric with scoring guide
- Runner usage and output structure
- Interpretation guide for results

### 4. Research Index Update

**File**: `research_agent/docs/research_index.md`

Added PDB entry to the top of Benchmark Evaluations section, positioned before NRB and Research Supervisor Optimizations.

## Validation Results

✓ All 5 case JSON files load and validate
✓ Case structure has all required fields: id, question, difficulty, project_state, ground_truth
✓ Runner can load cases with filters (--list works)
✓ Import of shared utilities (run_claude, parse_eval_json) from existing runner successful

## Smoke Test Status

Running `run_project_benchmark.py --case pdb_001`:
1. Single agent condition — in progress
2. Debate system condition — pending (will call research-supervisor agent)
3. Evaluator condition — pending

Expected output:
- `results/pdb_run_YYYY-MM-DD_HHMM/pdb_001_single.md` (2-5KB)
- `results/pdb_run_YYYY-MM-DD_HHMM/pdb_001_debate.md` (10-20KB)
- `results/pdb_run_YYYY-MM-DD_HHMM/pdb_001_eval_raw.md` (2-5KB)
- `results/pdb_run_YYYY-MM-DD_HHMM/pdb_001_eval.json` (structured rubric scores)

## Files Created

```
research_agent/benchmarks/
├── run_project_benchmark.py            [NEW - 390 lines]
└── project_cases/
    ├── pdb_001.json
    ├── pdb_002.json
    ├── pdb_003.json
    ├── pdb_004.json
    └── pdb_005.json

research_agent/docs/
├── project_decision_benchmark.md       [NEW]
└── research_index.md                   [UPDATED - added PDB entry]
```

## Next Steps

1. **Complete smoke test** — Verify pdb_001 runs end-to-end without errors
2. **Run all 5 cases** — If smoke test passes, run full benchmark
3. **Analyze results** — Compare per-criterion lifts (identify which advantages debate shows)
4. **Document findings** — Write results summary linking to debate history patterns

## Design Rationale

The PDB bridges the gap between the NRB (academic benchmark) and real project needs:

- **Size**: Larger project state (~2K-4K words) tests context distribution value
- **History**: Ground truth includes confounds + prior decisions, testing institutional memory
- **Rubric**: Decomposed criteria let us pinpoint where debate adds value (design ≠ insight recovery per NRB)
- **Retrospective**: Uses project's own debates, avoiding contamination risk and ensuring real-world relevance
- **Protocol**: Enforces the supervisor's 7-step process, testing forced sequencing + kill criteria

Expected to show: Debate > single on Confound Detection, Experiment Sequencing, History Anti-Repetition. Single ≈ debate on Context Utilization (both see same data).
