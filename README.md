# RepoBrain

RepoBrain is a local-first documentation generator for software repositories. It scans a Git
repository, parses its source code into a structured intermediate representation (IR), analyzes
class/package dependencies and method-call flows, and uses a **local** LLM (via
[Ollama](https://ollama.com), e.g. Qwen3) to generate `README.md`, `ARCHITECTURE.md`, and
`SEQUENCE.md`.

No source code ever leaves your machine: the LLM only ever sees structured summaries (class
names, signatures, doc comments, dependency edges) that RepoBrain extracts from the code — never
raw source text — and it talks exclusively to a local Ollama daemon.

This is v1: a documentation pipeline, not a chatbot. It supports Java today; the parser and LLM
provider layers are both pluggable so other languages/models can be added without touching the
rest of the pipeline (see [Extending RepoBrain](#extending-repobrain)).

## How it works

```
 Git repo
    │  1. scan (git ls-files, respects .gitignore)
    ▼
 RepoScanner ──────────────────────────────────────┐
    │  2. parse (tree-sitter)                      │
    ▼                                               │
 LanguageParser (JavaParser) ─▶ FileIR / ClassInfo  │  repobrain/scanner
    │  3. build symbol index, dependency graph,     │  repobrain/parsing
    │     and call graph                            │  repobrain/ir
    ▼                                               │
 SymbolIndex + DependencyGraph + call edges         │  repobrain/analysis
    │  4. build structured context (no raw source)  │
    ▼
 ProjectContext ─▶ prompts ─▶ LLMProvider (Ollama)  │  repobrain/docgen
    │  5. write docs, only if structural fingerprint changed
    ▼                                               │  repobrain/llm
 docs/generated/{README,ARCHITECTURE,SEQUENCE}.md
```

### Pipeline stages

1. **Scanning** (`repobrain/scanner/repo_scanner.py`) — lists every non-ignored file in the
   repository via `git ls-files`, so `.gitignore` is respected for free. RepoBrain requires the
   target to be a Git working tree (this is also what makes incremental updates possible).
2. **Parsing** (`repobrain/parsing/`) — each file is parsed by a `LanguageParser` implementation
   using [Tree-sitter](https://tree-sitter.github.io/tree-sitter/). `JavaParser` walks the concrete
   syntax tree and extracts package, imports, classes/interfaces/enums/records, fields, methods,
   modifiers, Javadoc, and each method's body call sites (receiver + method name, in source order)
   — into the language-agnostic IR defined in `repobrain/ir/models.py` (`FileIR`, `ClassInfo`,
   `MethodInfo`, `MethodCall`, …).
3. **Analysis** (`repobrain/analysis/`) —
   - `symbol_extractor.py` flattens the parsed repo into a `SymbolIndex` (lookup by qualified name,
     simple name, and package).
   - `dependency_analyzer.py` resolves type references (`extends`, `implements`, field/parameter/
     return types, `new X()`) into a `DependencyGraph` between classes actually defined in the
     repo — JDK/third-party types are excluded from the graph but summarized separately as a
     "tech stack" signal.
   - `call_graph.py` resolves each method call's *receiver* (a field or parameter name) to its
     declared type, then to a project class, giving a call graph of `(class, method) -> (class,
     method)` edges — the basis for sequence diagrams. Calls on local variables, chained
     expressions (`a.b().c()`), or external/JDK types don't resolve and are dropped rather than
     guessed.
4. **Documentation generation** (`repobrain/docgen/`) —
   - `context.py` renders the index + dependency graph into compact `ClassCard`/`ProjectContext`
     summaries, and folds in `sequence.py`'s selected call flows — this structured summary is the
     only thing ever sent to the LLM.
   - `sequence.py` picks up to 8 "likely entry point" methods (public, not themselves called by
     other resolved project code, ranked by outgoing-call count), traces each one through the call
     graph (depth- and step-capped to stay readable), and renders each as a **deterministic**
     Mermaid `sequenceDiagram` block — the diagram itself is never left to the LLM to reproduce;
     only the prose describing it is. Participants are laid out left-to-right by architectural
     layer (API/controller → service → repository/database, inferred from package and class name
     keywords), not raw call order — so a service that touches its repository before calling
     another service still renders with the repository on the right.
   - `prompts.py` + `generators.py` turn that context into one prompt per document.
   - `fingerprints.py` hashes the *structural* shape relevant to each document (signatures/package
     edges for ARCHITECTURE.md, resolved call chains for SEQUENCE.md, etc.) so `repobrain update`
     can skip regenerating a document whose relevant structure didn't actually change.
5. **LLM call** (`repobrain/llm/`) — `OllamaProvider` posts to the local Ollama HTTP API
   (`/api/generate`, `/api/tags`), with `think: false` for Qwen3-style models to skip the
   `<think>` reasoning trace.
6. **Incremental updates** (`repobrain/cache/`, `repobrain/scanner/git_diff.py`,
   `repobrain/pipeline.py`) — a state file at `<repo>/.repobrain/state.json` caches the last
   processed commit, the merged repo IR, and per-document fingerprints. `repobrain update`:
   - diffs the working tree against the last processed commit (`git diff --name-status`),
   - also catches brand-new untracked files by comparing the live scan against the cache (Git
     diff never reports untracked files),
   - re-parses only the changed/added files (skipping re-parse entirely if the file's content
     hash is actually unchanged, e.g. a stale dirty flag from an uncommitted edit),
   - drops removed files from the cached IR,
   - recomputes fingerprints and regenerates **only** the documents whose fingerprint changed —
     e.g. editing a method body without changing what it calls leaves SEQUENCE.md untouched.

## Project layout

```
repobrain/
  cli.py                  CLI entry point (generate / update / scan)
  config.py               YAML config loading + dataclasses
  logging_setup.py         logging configuration
  pipeline.py              top-level orchestrator (full + incremental runs)
  default_config.yaml      packaged defaults
  ir/models.py             language-agnostic intermediate representation
  scanner/                 file discovery (Git-aware) + git diff-based change detection
  parsing/                 LanguageParser interface, JavaParser (tree-sitter), registry
  analysis/                symbol index, dependency graph, and call graph construction
  llm/                     LLMProvider interface, OllamaProvider, registry
  docgen/                  prompt context, sequence diagrams, prompts, generators, fingerprints
  cache/                   on-disk incremental-update state store
config/
  repobrain.example.yaml   starting point for a project-local override file
tests/                     pytest suite (fixtures/sample_java_repo is a tiny 4-class Java project)
```

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) running locally, and a pulled model
(default `qwen3:8b`).

```bash
# 1. Install Ollama and pull a model (one-time)
ollama pull qwen3:8b

# 2. Set up RepoBrain
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Full analysis + documentation generation
repobrain generate /path/to/some/java/repo

# Re-analyze after making changes, regenerating only affected docs
repobrain update /path/to/some/java/repo

# Parse-only dry run (no LLM calls) — sanity-check parsing/structure first
repobrain scan /path/to/some/java/repo
```

Generated docs land in `<repo>/docs/generated/` by default. Useful flags (all three subcommands
accept them): `--model`, `--host`, `--language`, `--output-dir`, `--config <path>`, `--log-level`.
`generate`/`update` also accept `--force` to regenerate every document regardless of fingerprint.

Copy `config/repobrain.example.yaml` into a target repo and pass `--config` to override defaults
(exclude patterns, which docs to generate, LLM host/model/temperature, etc.) without CLI flags.

## Running the tests

```bash
pytest
```

The suite (90 tests) covers the scanner, the Java parser (generics, nested classes, enums,
records, malformed source, call-site extraction), the dependency graph, the call graph and
sequence-diagram selection/tracing/layer-ordering, git-diff-based change detection, the
fingerprint/skip logic, and the CLI — all against a fake, deterministic `LLMProvider` so it never
needs a running Ollama daemon. `tests/fixtures/sample_java_repo` is a tiny hand-written 4-class
Java project used throughout.

## Extending RepoBrain

- **New language**: implement `LanguageParser` (see `repobrain/parsing/java_parser.py`) and
  register it in `repobrain/parsing/registry.py`. Nothing else changes — scanning, analysis, and
  doc generation all operate on the shared IR.
- **New LLM provider**: implement `LLMProvider` (see `repobrain/llm/ollama_provider.py`) and
  register it in `repobrain/llm/registry.py`. Select it via `llm.provider` in config.
- **New generated document**: add a prompt builder in `repobrain/docgen/prompts.py`, a small
  `DocGenerator` subclass in `repobrain/docgen/generators.py`, and a fingerprint entry in
  `repobrain/docgen/fingerprints.py`.

## Known limitations (v1)

- Java only; the IR and pipeline are language-agnostic but only `JavaParser` exists today.
- Large repositories are truncated to fit a character budget derived from `llm.num_ctx`
  (`char_budget_for_num_ctx` in `repobrain/docgen/context.py`), split evenly across packages so no
  single large package starves the rest — with a hard ceiling of `MAX_CLASSES_IN_DETAIL` (150)
  classes regardless. Beyond that, package summaries still get accurate counts, but per-class
  detail is omitted; per-class method/field lists are also capped individually so one huge class
  can't blow the budget on its own. If you see `ctx.truncated` / a "more detail than fits" log
  warning, raise `llm.num_ctx` (and `llm.timeout_seconds` to match) for more detail per doc.
  Chunked/hierarchical summarization for very large repos is a natural next step.
- Dependency resolution is conservative (unambiguous name matches only); ambiguous simple-name
  collisions across packages are silently dropped rather than guessed.
- Call resolution (for SEQUENCE.md) only tracks receivers that are fields or method parameters —
  a call on a local variable (`Widget w = make(); w.getName();`) won't resolve, since local
  variable types aren't tracked. This undercounts flows in code that assigns to locals before
  calling, but avoids guessing wrong. Entry-point selection is a heuristic ("public, not called by
  other resolved project code, most outgoing calls") — it's a reasonable proxy for "meaningful
  workflow," not a guarantee; some genuinely important flows may not surface.
