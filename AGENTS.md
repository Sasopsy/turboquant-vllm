# AGENTS.md

## Scope

This repository is a design-doc workspace for an out-of-tree TileLang TurboQuant-style KV-cache integration with `vLLM`.

The repo is about:

- architecture and component design
- older-branch `vLLM` compatibility
- plugin-owned integration points
- execution, memory, and metadata correctness

The repo is not the implementation itself. Do not turn design docs into speculative code plans unless explicitly asked.

## Read This First

When starting work, review these in order:

1. [ARCHITECURE.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/ARCHITECURE.md)
2. the newest relevant file in [SESSION_LOGS/](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/SESSION_LOGS)
3. [COMPONENTS/C1-variant-registry-and-config.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS/C1-variant-registry-and-config.md)
4. [COMPONENTS/C2-offline-precomputation.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS/C2-offline-precomputation.md)
5. [COMPONENTS/C3-quantization-config-and-kv-cache-method.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS/C3-quantization-config-and-kv-cache-method.md)
6. [COMPONENTS/C4-C5-backend-metadata-and-impl.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS/C4-C5-backend-metadata-and-impl.md)
7. [COMPONENTS/C6-C7-kernels-and-plugin-registration.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS/C6-C7-kernels-and-plugin-registration.md)

Notes:

- The filename `ARCHITECURE.md` is intentionally misspelled in-repo; preserve the existing filename unless explicitly asked to rename it.
- Session history is folder-based now. Prefer the most recent log first, then pull older logs only if the current task needs extra context.

## Canonical Sources

Use these as the source of truth:

- [ARCHITECURE.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/ARCHITECURE.md)
  - top-level design and end-to-end invariants
- [COMPONENTS/](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS)
  - canonical low-level design docs

Use these as reference material only:

- [extra-docs/agent-hld-guide](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/agent-hld-guide)
- [extra-docs/agent-findings-vllm](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/agent-findings-vllm)
- [extra-docs/deep-dive-understanding](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/deep-dive-understanding)

If a reference doc disagrees with `ARCHITECURE.md` or `COMPONENTS/`, prefer the canonical docs and update the reference later if needed.

## Current Design Posture

Preserve these assumptions unless the user explicitly asks to change them:

- Target compatibility includes older `vLLM` branches, not just the newest tree.
- Upstream TurboQuant in a local `vLLM` checkout is reference material, not an assumed dependency.
- Plugin-owned identifiers remain canonical:
  - `quantization="tq_3bit"`
  - `quantization="tq_4bit"`
  - `kv_cache_dtype="tilelang_tq_3bit"`
  - `kv_cache_dtype="tilelang_tq_4bit"`
- Both keys and values stay on the TurboQuant MSE + QJL path.
- Do not revisit the core algorithm unless the user explicitly asks for it.

## Compatibility Invariants

These are easy to break and should be treated as hard constraints:

- Split KV-store work must use `slot_mapping.shape[0]` as the authoritative row bound.
- Negative `slot_mapping` entries are no-store sentinels.
- `num_actual_tokens` is not the authoritative store bound in padded split-KV execution.
- No inference-critical GPU allocation may first appear after `profile_run()`.
- Offline artifacts may originate on CPU, but runtime-resident device buffers must exist before profiling.
- Docs must not assume upstream `vLLM` natively accepts custom cache-dtype literals on older branches.
- Branch-specific shims/adapters are part of the design story whenever upstream hooks differ.

## Repo Layout

- [ARCHITECURE.md](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/ARCHITECURE.md)
  - architecture doc
- [COMPONENTS/](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/COMPONENTS)
  - numbered component docs
- [SESSION_LOGS/](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/SESSION_LOGS)
  - timestamped session summaries and handoff context
- [extra-docs/agent-hld-guide](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/agent-hld-guide)
  - guide material used during design review
- [extra-docs/agent-findings-vllm](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/agent-findings-vllm)
  - `vLLM` integration findings
- [extra-docs/deep-dive-understanding](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/extra-docs/deep-dive-understanding)
  - explanatory notes and learning-oriented deep dives

## Editing Rules

- Keep numbering, filenames, titles, and cross-references aligned.
- If a component number changes, update:
  - filename
  - document title
  - dependency lines
  - cross-doc references
  - any session-log references that would become misleading
- Preserve the distinction between:
  - algorithm choices
  - `vLLM` integration mechanics
  - branch compatibility shims
- Avoid silently introducing upstream-specific assumptions into canonical docs.
- When unsure whether something is architectural or branch-specific, frame it explicitly rather than blending the two.

## Session Log Workflow

Always use [SESSION_LOGS/](/Users/sasmitdatta/Desktop/turboquant-vllm-tl/SESSION_LOGS) as lightweight working memory for the repo.

At the start of a task:

- read the most recent relevant log first
- use older logs only if the newest one does not contain enough context
- extract only the context that matters for the current task
- summarize prior work mentally instead of dragging full historical detail into the docs

When choosing logs:

- prefer lexicographically newest timestamped files
- prefer logs that mention the same files or subsystem you are editing
- avoid loading the entire folder unless the task explicitly needs a long history

After meaningful work:

- add a new timestamped session log file when the work materially changes repo context
- use a filename like `YYYY-MM-DD_HH-MM-SS_TZ_short-topic.md`
- keep each log self-contained but concise
- include what changed, why it changed, and any follow-up risks or open questions

Good session log content:

- renamed docs or folders
- changed numbering or file layout
- compatibility decisions
- resolved review findings
- open issues that matter for the next editor

Bad session log content:

- full chat transcript
- low-signal tool chatter
- repeated restatement of stable architecture context

Preferred session log shape:

- timestamp
- scope
- key changes
- important decisions/invariants
- open items

## Expected Behavior For Future Agents

- Start from the current repo state, not stale assumptions from earlier sessions.
- Check whether filenames and directories changed before editing links.
- Use the component docs and architecture doc to keep language consistent.
- Update `SESSION_LOGS/` when the work changes project context in a way a later agent would need to know.
- Keep summaries brief, contextual, and useful for handoff.
