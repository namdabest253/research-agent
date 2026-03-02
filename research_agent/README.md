# Research Agent — Multi-Agent Debate System for Claude Code

A multi-agent research system built entirely on [Claude Code](https://docs.anthropic.com/en/docs/claude-code) native agents. No Python orchestration — agents are markdown files that Claude Code launches via its `Task` tool.

**What it does:** Searches arXiv and Semantic Scholar for papers, reads and analyzes them, maintains a structured knowledge base, and orchestrates structured debates between specialist agents to make better research decisions than any single perspective can produce.

---

## How It Works

### Research Phase

```
You: "Find papers on discrete diffusion for 3D generation"
  │
  ▼
Claude Code spawns ──→ research-paper-finder agent (Sonnet)
                          │
                          ├── python3 research_agent/tools.py search "..."
                          ├── python3 research_agent/tools.py read "2303.12345"
                          ├── python3 research_agent/tools.py citations "..."
                          │
                          ▼
                       docs/[topic]_papers.md          (full detail)
                       docs/debate_briefs/[topic].md   (condensed, advocate-targeted)
                       docs/debate_briefs/brief_index.md
```

### Debate Phase

```
You: "Should we use discrete diffusion or continuous latent diffusion?"
  │
  ▼
research-supervisor (Opus)
  │
  ├── STEP 0: Load context
  │     ├── STATUS.md (~35 lines — current state)
  │     ├── docs/debates/index.md (~20 lines — past decisions, tags, outcomes)
  │     └── [optionally] context-retriever for targeted extracts
  │
  ├── STEP 1: Frame the debate question precisely
  │
  ├── STEP 1.5: Pre-round — all 3 advocates identify KEY SUB-QUESTIONS (parallel)
  │
  ├── STEP 2: Round 1 — independent positions (3 advocates, parallel)
  │     ├── Empiricist  → data-driven position citing specific metrics
  │     ├── Theorist    → first-principles position with mechanism
  │     └── Contrarian  → surfaces blind spots, proposes concrete alternative
  │
  ├── STEP 3: Round 2 — cross-examination (3 advocates, parallel)
  │     └── each reads all Round 1 positions, challenges the weakest, updates if warranted
  │
  ├── STEP 4: Synthesis
  │     ├── points of genuine agreement
  │     ├── points of genuine disagreement (needs human judgment or more evidence)
  │     └── decision with traceability tags [from Empiricist R2], [from Theorist R1], etc.
  │
  ├── STEP 5: Falsification
  │     └── research-falsifier → GO / CONDITIONAL GO / NO-GO + kill criteria
  │
  ├── STEP 5.5: Overconfidence calibration
  │     └── all 3 advocates give failure-probability estimates (0–100%)
  │         sum < 60% → OVERCONFIDENCE WARNING
  │         sum > 250% → LOW VIABILITY WARNING
  │
  └── STEP 6: Write decision to docs/debates/YYYY-MM-DD_slug.md
                └── update docs/debates/index.md
```

---

## Agent Architecture

All agents live in [`.claude/agents/`](.claude/agents/). Claude Code reads these markdown files as system prompts when spawning sub-agents.

### Core Agents (invoke directly)

| Agent | Model | Role |
|-------|-------|------|
| **research-paper-finder** | Sonnet | Searches, reads, and analyzes papers. Generates debate briefs. |
| **research-supervisor** | Opus | Orchestrates debates. Invokes advocates, falsifier, context-retriever. |

### Debate Advocate Sub-Agents (spawned by supervisor — do not invoke directly)

| Agent | Model | Epistemology |
|-------|-------|-------------|
| **research-advocate-empiricist** | Sonnet | Only accepts measured results, ablations, convergence data. Every claim must cite a metric. Follows DMAD procedure: collect data → identify patterns → test against baselines → derive recommendation. |
| **research-advocate-theorist** | Sonnet | Only accepts math, loss landscape analysis, information-theoretic bounds. Rejects "it worked" without a mechanism. |
| **research-advocate-contrarian** | Sonnet | Surfaces blind spots and unconsidered alternatives. Must commit to a specific alternative, not just critique. |

### Post-Debate Sub-Agents (spawned by supervisor)

| Agent | Model | Role |
|-------|-------|------|
| **research-falsifier** | Sonnet | Stress-tests the winning proposal. Identifies likely failure modes, designs kill criteria, issues GO/CONDITIONAL/NO-GO. |
| **context-retriever** | Haiku | Takes a natural-language query, reads key files, returns a focused extract with citations (~80 lines max). Does NOT reason — retrieves only. Used by supervisor in Step 0 to avoid reading full files. |

### Meta-Research Agents (optional — debates about the debate system itself)

| Agent | Role |
|-------|------|
| **meta-research-supervisor** | Same debate structure, but applied to improving the debate system |
| **meta-advocate-empiricist/theorist/contrarian** | Meta-debate advocates |
| **meta-research-falsifier** | Stress-tests proposed changes to the debate system |

---

## Directory Structure

This is the full recommended layout for your project. The agents expect these paths and file formats.

```
your-project/
  CLAUDE.md                           # Root config for Claude Code (sets project context)
  STATUS.md                           # ← CRITICAL: single source of truth for current state
  .claude/
    agents/                           # Agent definitions (markdown files)
      research-paper-finder.md
      research-supervisor.md
      research-advocate-empiricist.md
      research-advocate-theorist.md
      research-advocate-contrarian.md
      research-falsifier.md
      context-retriever.md
      [meta-agents if needed]
    agent-memory/
      research-supervisor/
        MEMORY.md                     # Supervisor's persistent memory (auto-updated)
  research_agent/
    tools.py                          # CLI (search, read, citations, knowledge base)
    knowledge_base.json               # Structured paper storage
    requirements.txt
    CLAUDE.md                         # Research agent conventions
    docs/
      research_index.md               # Index of all topic files
      metrics_registry.md             # ← Structured tables of all measured results
      debates/
        index.md                      # ← 1-line-per-debate table (supervisor reads this)
        YYYY-MM-DD_slug.md            # Individual debate decisions (read selectively)
      debate_briefs/
        brief_index.md                # Maps topics → brief files
        [topic]_brief.md              # Condensed advocate-targeted brief (<200 lines)
      [topic]_papers.md               # Full research output per topic
    .cache/                           # Cached paper text extractions (safe to clear)
```

---

## Key Files: What Each One Does and Who Writes It

### `STATUS.md` (project root)
**Written by:** You, after each experiment or debate.
**Read by:** Every agent, first, in every session.

The single source of truth for what is happening right now. Agents read this before anything else so they understand project state without parsing prose from multiple files.

```markdown
# Project Status

## Current Phase
**Phase X.Y — [name]** (one-line description)

## Active Experiment
What is being tested right now

## Hypothesis Being Tested
What we expect to learn

## Last Experiment Result
**[Experiment name] [PASSED/FAILED]** (date) — one-line result

## Next Actions
1. First thing to do
2. Second thing to do

## Key Numbers (latest that matter)
- Metric A: value
- Metric B: value

## Confirmed Dead Ends
- Approach X: reason it failed — do not retry
- Approach Y: reason it failed — do not retry

*Last updated: YYYY-MM-DD*
```

Keep it under 40 lines. If it grows longer, you're putting the wrong things in it — move detail to `docs/` or `CLAUDE.md`.

---

### `docs/debates/index.md`
**Written by:** research-supervisor (after each debate, in Step 6).
**Read by:** research-supervisor (Step 0, always), advocates (when needed).

A compact table — one row per debate. The supervisor reads this (20 lines) instead of the full debate history. Individual debate files are read only when their tags match the current question.

```markdown
| # | Date | Title | Outcome | Tags | File |
|---|------|-------|---------|------|------|
| 1 | 2026-02-25 | v7 Approach | MSE-only + rotation aug | diffusion, training | [→](./2026-02-25_v7_approach.md) |
| 2 | 2026-02-26 | Discrete Pivot | Path C (full voxel) not Path B | discrete, pivot | [→](./2026-02-26_discrete_pivot.md) |
```

**Tags** are the key to selective reading. Use domain-relevant tags that match the vocabulary of your debate questions. If a new debate is about discrete diffusion, the supervisor reads only files tagged `discrete` or `diffusion`, not the full history.

---

### `docs/metrics_registry.md`
**Written by:** You, after each experiment.
**Read by:** context-retriever (on demand), Empiricist advocate (directly if needed).

Structured tables of every measured result. Gives the Empiricist a single queryable source instead of having to parse prose across multiple files.

```markdown
# Metrics Registry

## [Component] — Measured Results

| Version | Key Metric | Secondary Metric | Notes |
|---------|------------|-----------------|-------|
| v1      | 0.305      | ~25% partial    | Baseline |
| v2      | —          | 0/N             | Failed: overfitting |
```

Include: versions/variants tested, their key metrics, train/val gaps if relevant, and a one-phrase outcome. Entries that say "FAILED" with a reason are as valuable as successes — the Empiricist uses them as baselines.

---

### `docs/debate_briefs/[topic]_brief.md`
**Written by:** research-paper-finder.
**Read by:** research-supervisor (to inject into advocate prompts).

Condensed summaries of research literature, structured by advocate. Each brief has three sections that the supervisor uses to give each advocate only their relevant slice:

```markdown
## TL;DR
[2-3 sentence overview of what this literature says]

## Techniques Catalog
[table of methods + key results]

## For the Empiricist
[metrics, benchmarks, ablation results — things with numbers]

## For the Theorist
[mathematical properties, loss landscape analysis, theoretical guarantees]

## For the Contrarian
[limitations, failure modes, alternatives, what these papers don't address]
```

This solves the training-data-cutoff problem: the paper-finder discovers new techniques via web search, distills them into briefs, and the supervisor injects those briefs so advocates can reason about research they wouldn't otherwise know about.

---

### `CLAUDE.md` (root)
**Written by:** You.
**Read by:** Claude Code (always loaded as project context).

Project overview for Claude Code sessions. Should include:
- What the project does (1-2 sentences)
- Directories to IGNORE (binary data, logs, caches — prevents agents exploring noise)
- Pointer to STATUS.md and QUICKSTART.md
- Directory map (relevant paths only)
- List of available agents and what they're for

---

## Setting Up for Your Own Project

### Step 1: Copy the agent files

Copy everything in `.claude/agents/` to your project's `.claude/agents/`. The agent markdown files are the entire system — they contain the debate protocol, reasoning procedures, and output formats.

### Step 2: Create STATUS.md

Create a `STATUS.md` at your project root using the format above. Fill it in with your current project state. This is what agents will read first in every session.

### Step 3: Create the debates/ directory

```bash
mkdir -p research_agent/docs/debates
```

Create `research_agent/docs/debates/index.md` with the table header (empty initially):

```markdown
# Debate Index

| # | Date | Title | Outcome | Tags | File |
|---|------|-------|---------|------|------|
```

### Step 4: Create metrics_registry.md

Create `research_agent/docs/metrics_registry.md` with your project's measured results. If you're just starting, create the file with empty tables for the metrics you plan to track. Update it after every experiment.

### Step 5: Configure CLAUDE.md

Create a root `CLAUDE.md` that:
- Points agents to STATUS.md ("Start here: STATUS.md")
- Lists directories to ignore (your equivalent of binary data, large datasets, etc.)
- Maps the relevant directory structure

### Step 6: Install Python dependencies

```bash
pip install -r research_agent/requirements.txt
```

Dependencies: `arxiv`, `pymupdf` (fitz), `httpx`.

### Step 7: Run Claude Code from the project root

```bash
cd your-project/
claude
```

The agents are automatically available. Invoke them by name:
- "Find papers on [topic]" → research-paper-finder runs
- "Debate whether [X] or [Y]" → research-supervisor runs

---

## CLI Tools

The agents call `tools.py` via Bash. You can also use them standalone:

```bash
# Search arXiv
python3 research_agent/tools.py search "VQ-VAE 3D generation" --max-results 5

# Search Semantic Scholar
python3 research_agent/tools.py search-ss "latent diffusion 3D" --max-results 10 --year "2023-2024"

# Read a paper (downloads PDF, extracts text, caches)
python3 research_agent/tools.py read "2303.12345"
python3 research_agent/tools.py read "2303.12345" --pages "4-8"

# Citation graph traversal
python3 research_agent/tools.py citations "2107.03006" --direction both --max-results 20

# Knowledge base
python3 research_agent/tools.py kb add --paper-id "2107.03006" --title "Title" --tags "diffusion,3D" --relevance-score 8.5
python3 research_agent/tools.py kb search --query "diffusion"
python3 research_agent/tools.py kb list
python3 research_agent/tools.py kb export
```

Cache is keyed by arXiv ID. Re-reading a cached paper is instant. Clear `.cache/` if disk space is needed.

---

## How the Debate System Prevents Bad Decisions

### Three incompatible epistemologies force completeness

The advocates are designed so that no single perspective can dominate without addressing the others:

- **Empiricist** grounds everything in data — no handwaving. "This should work theoretically" is rejected unless accompanied by a measured result from an analogous system.
- **Theorist** demands mechanistic understanding — "it worked" is not enough. Every recommendation needs a reason why it works, grounded in math or loss landscape analysis.
- **Contrarian** prevents groupthink — must propose a *specific* concrete alternative, not just criticize. This forces the debate to consider the full option space.

### The pre-round (Step 1.5) aligns advocates before they diverge

Before Round 1 positions are stated, all three advocates answer: "What are the 3 most important sub-questions to answer before committing to a recommendation?" The supervisor synthesizes 3–5 key sub-questions, then injects them into every Round 1 prompt. This prevents advocates from talking past each other.

### Traceability prevents supervisor bias

Every element of the synthesis must be tagged `[from Empiricist R2]`, `[from Theorist R1]`, etc. Any element the supervisor adds from its own reasoning must be tagged `[supervisor addition]` and justified. This makes it visible if the supervisor is injecting its own preferences rather than faithfully representing the debate.

### Overconfidence calibration catches false consensus

After falsification, all three advocates independently estimate the probability (0–100%) that the recommendation will **fail**. If the three estimates sum to less than 60%, the system flags an `OVERCONFIDENCE WARNING` — the debate may have produced false consensus. Using failure-probability framing (not success) is deliberate: LLMs are better calibrated when reasoning about failure modes.

### The Falsifier designs kill criteria, not just warnings

The Falsifier's job is not to find problems — it's to design the *cheapest pre-test* that would catch a fatal flaw before committing to a multi-week implementation. Every conditional pass comes with specific monitoring criteria: what to measure, at what epoch, and what threshold triggers an abort.

---

## Maintaining the System Over Time

### After every experiment
1. Update `STATUS.md` with the result and new next actions
2. Add a row to `docs/metrics_registry.md` with the measured outcome

### After every debate
The supervisor handles this automatically (Step 6). It writes a new file to `docs/debates/` and updates `docs/debates/index.md`.

### When the debate index gets long (>15 entries)
Consider adding a "Project" column to the index to distinguish between different research tracks if your project has diverged into separate sub-problems.

### When a dead end is confirmed
Add it to the "Confirmed Dead Ends" section of `STATUS.md`. Remove approaches from your active consideration. The Empiricist's DMAD procedure uses these as baselines — known failures are as valuable as successes.

---

## Other Tools

| Path | What |
|------|------|
| `research_agent/validate_debate.py` | Schema validator for debate log entries |
| `research_agent/tests/` | Tests for the debate validator |
| `research_agent/benchmarks/` | Benchmark suite for evaluating debate quality |
| `research_agent/debate_viz/` | HTML visualization for debate flows |
