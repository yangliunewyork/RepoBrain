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
   target to be a Git working tree (this is also what makes incremental updates possible). Test
   sources are excluded by default (`src/test/**`, `*Test.java`, `*IT.java`, ... — mirroring Maven
   Surefire/Failsafe's own conventions) so a test method with no other callers never gets mistaken
   for a real application entry point in SEQUENCE.md.
2. **Parsing** (`repobrain/parsing/`) — each file is parsed by a `LanguageParser` implementation
   using [Tree-sitter](https://tree-sitter.github.io/tree-sitter/). `JavaParser` walks the concrete
   syntax tree and extracts package, imports, classes/interfaces/enums/records, fields, methods,
   modifiers, annotations (class- and method-level, e.g. `@RestController`, `@GetMapping`, `@Test`),
   Javadoc, and each method's body call sites (receiver + method name, in source order) — into the
   language-agnostic IR defined in `repobrain/ir/models.py` (`FileIR`, `ClassInfo`, `MethodInfo`,
   `MethodCall`, …).
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
   - `layers.py` classifies each class as `api`/`service`/`domain`/`data`/unclassified. Framework
     annotations (`@RestController`/`@Controller`/`@*Mapping` → api, `@Service` → service,
     `@Entity`/`@Table`/`@Document`/`@Embeddable` → domain, `@Repository`/`@Dao` → data) are checked
     first — a verified signal, not a guess — then a name-keyword fallback (`*Controller`,
     `*Repository`, ...), then a structural fallback for domain types with no annotation at all (a
     class/record with an instance field, or any enum). Also aggregates the class-level dependency
     graph into a small **component graph** (`component_graph`) — API/Service/Domain/Data/Other
     nodes plus a node per detected external system — a far more legible "architecture" view than
     one node per Java package.
   - `external_systems.py` classifies real import statements against known library prefixes
     (JDBC/JPA, NoSQL/cache, messaging, HTTP clients, cloud SDKs, email) into categories like
     "Relational Database" or "Cloud Provider SDK" — grounding ARCHITECTURE.md's "what external
     systems does this talk to" answer in actual imports rather than a guess.
4. **Documentation generation** (`repobrain/docgen/`) —
   - `context.py` renders the index + dependency graph into compact `ClassCard`/`ProjectContext`
     summaries — including each class's verified layer and a repo-wide layer breakdown — and folds
     in `sequence.py`'s selected call flows. This structured summary is the only thing ever sent
     to the LLM.
   - `sequence.py` selects up to 8 flows, preferring (1) methods with an explicit route/handler
     annotation (`@GetMapping`, `@RequestMapping`, JAX-RS `@GET`/`@Path`, ...) — a confirmed entry
     point, not a heuristic; then (2) public methods not themselves called by other resolved
     project code; ranked by outgoing-call count. Methods annotated `@Test`/`@ParameterizedTest`/
     etc. are never candidates, as a second layer of defense beyond file-level exclusion. Each
     flow is traced through the call graph (depth- and step-capped to stay readable) and rendered
     as a **deterministic** Mermaid `sequenceDiagram` — the diagram itself is never left to the
     LLM to reproduce, only the prose describing it is. Participants are laid out left-to-right by
     `layers.py`'s classification (API → service → data), not raw call order — so a service that
     touches its repository before calling another service still renders with the repository on
     the right.
   - `prompts.py` + `generators.py` turn that context into one prompt per document.
     ARCHITECTURE.md's prompt is organized entirely around responsibility and runtime behavior,
     never around package layout: it states the verified layer breakdown, external systems, and
     domain model as ground truth, includes the "primary request flow" (the highest-confidence
     sequence flow — see below), and instructs the model to answer six questions explicitly as
     sections — what the service does, its business capabilities, its runtime components, the
     external systems it talks to, where data enters/exits, and its primary request flow — rather
     than describing packages.
   - `fingerprints.py` hashes the *structural* shape relevant to each document (full per-class
     shape + layer breakdown + external systems + primary flow for ARCHITECTURE.md, resolved call
     chains for SEQUENCE.md, etc.) so `repobrain update` can skip regenerating a document whose
     relevant structure didn't actually change.
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
  analysis/                symbol index, dependency graph, call graph, layer/domain classification,
                           component graph, and external-system detection
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

The suite (133 tests) covers the scanner (including default test-source exclusion and a
regression test for a glob-matching edge case in top-level directory excludes), the Java parser
(generics, nested classes, enums, records, malformed source, call-site extraction, annotation
extraction), the dependency graph, layer/domain classification and the component graph, external-
system detection, the call graph and sequence-diagram selection/tracing/annotation-aware entry
points/layer-ordering, git-diff-based change detection, the fingerprint/skip logic, and the CLI —
all against a fake, deterministic `LLMProvider` so it never needs a running Ollama daemon.
`tests/fixtures/sample_java_repo` is a tiny hand-written 4-class Java project used throughout.

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
  calling, but avoids guessing wrong.
- Entry-point selection prefers a verified route/handler annotation when one is present, but falls
  back to a heuristic ("public, not called by other resolved project code, most outgoing calls")
  for everything else — a reasonable proxy for "meaningful workflow," not a guarantee; some
  genuinely important flows in unannotated code may not surface.
- Layer classification (`analysis/layers.py`) recognizes Spring (`@RestController`, `@Controller`,
  `@*Mapping`, `@Service`, `@Repository`, `@Entity`, `@Table`, `@Document`) and JAX-RS (`@Path`,
  `@GET`/`@POST`/...) annotations today; other frameworks fall back to the name-keyword heuristic,
  which is weaker. Extending the recognized annotation sets in `layers.py` (and the route/test
  annotation sets in `docgen/sequence.py`) is the natural way to support another framework.
- The domain-model structural fallback (any class/record with an instance field, or any enum,
  once annotation and keyword checks come up empty) is intentionally broad — DTOs, request/
  response objects, and config-holder classes with plain fields will also get classified as
  "domain" if they don't match anything more specific first. This trades some over-inclusion for
  never missing a real domain type; a class made entirely of static members is the one shape
  reliably excluded.
- External-system detection (`analysis/external_systems.py`) only recognizes a fixed list of
  common JVM ecosystem library prefixes (JDBC/JPA/Hibernate, MongoDB/Redis, Kafka/RabbitMQ,
  OkHttp/Retrofit/Apache HttpClient, AWS/GCP/Azure SDKs, JavaMail). An integration using something
  else — an in-house client library, a less common driver — won't be detected; extending
  `_CATEGORY_PREFIXES` is the way to add more.
- The LLM can still embellish beyond the grounded facts even when instructed not to (e.g.
  inferring a specific state-transition story from an enum's constant names alone) — the prompt
  explicitly warns against this for the domain model section, but an 8B local model won't catch
  every case. Review generated prose the same way you'd review the diagrams: the structure is
  verified, some of the narrative around it is the model's inference.
- Test-code exclusion is pattern-based (`src/test/**`, `*Test.java`, `*IT.java`, ...) plus an
  annotation-level backstop for `@Test`/`@ParameterizedTest`/etc. methods specifically; a test
  class using neither a recognized path/suffix convention nor a recognized test annotation could
  still slip through and appear in generated docs.
