# Aegis-X — Complete Repository Documentation — v3.1

> **Verified against live codebase — April 2026 (Second Pass — Zero False Positives).**
> Every class, method, threshold, and behaviour described below has been manually checked against the actual source files.
> Previous v2 stubs ("See original.allfiles.md") have been replaced with full documentation.
>
> **v3.1 changes** — Added verified bug list, corrected weight discrepancies, fixed threshold documentation, added security notes.

---

## ⚠️ Discrepancies Fixed vs v2 allfiles.md

| Item | Old (Wrong) | Actual (Correct) |
|---|---|---|
| `run_univfd` weight | 0.15 | **0.20** (registry.py line 85) |
| `run_xception` weight | 0.10 | **0.15** (registry.py line 86) |
| `run_sbi` weight | 0.18 | **0.20** (registry.py line 87) |
| Weight sum | 1.00 claimed | **1.12** actual (registry sums > thresholds.py) |
| `agent.py` structure | `_run_cpu_phase()` + `_run_gpu_phase()` | **`analyze()` with Segments A/B/C + Face Gate** |
| `SBI` tool status | "Unchanged from v1" | **Modified** — threshold imports, skip gate fix |
| rPPG weight | 0.06 (only) | **0.06 ensemble weight** + **0.35 gate weight** (different systems) |
| `downloads/download_new_weights.py` | Listed as present | **Does not exist** in current repo |
| `day18_test_results.txt` | Listed | **Does not exist** |
| `diagnostics_day14.py` | Listed | **Does not exist** |
| `requirements.txt` | Single file | **Split into** `requirements-main.txt` + `requirements-gpu.txt` |
| UnivFD VRAM loaded | ~1.8 GB | **GPU_VRAM_REQUIREMENTS says 0.6 GB** (pre-flight check); actual CLIP peak ~1.8 GB |

---

## ⚠️ Verified Bugs & Known Discrepancies (April 2026 Second Pass)

> These are confirmed, zero-false-positive issues found by line-by-line code audit.

### Critical Bugs (Will Cause Runtime Failures or Wrong Results)

| # | File:Line | Bug | Impact |
|---|---|---|---|
| 1 | `llm.py:100` | `logger` undefined — `NameError` on LLM timeout | Pipeline crashes on LLM timeout |
| 2 | `thresholds.py:121` | `RPPG_HAIR_OCCLUSION_VARIANCE = 0.25` (should be ~35.0) | False hair occlusion on every frame |
| 3 | `thresholds.py:120` | `RPPG_CARDIAC_BAND_MAX_HZ = 2.5` (README says 4.0) | Misses cardiac peaks > 150 BPM |
| 4 | `run_web.py:30` | Path traversal via unsanitized filename | Security vulnerability |
| 5 | `registry.py` vs `thresholds.py` | 7 tool weights mismatch between files | Early stopping uses different weights than ensemble |

### Configuration Inconsistencies

| Setting | `config.py` default | `thresholds.py` value |
|---|---|---|
| `AGENT_MAX_RETRIES` | 2 | 3 |
| `LLM_MAX_TOKENS` | 1024 | 512 |

### Documentation vs. Code Discrepancies

| Item | README says | Code actually uses | File:Line |
|---|---|---|---|
| Decisive threshold | `\|score - 0.5\| > 0.05` | `\|score - 0.5\| > 0.15` | `agent.py:168` |
| Gate decisive count | `< 2` → FULL_GPU | `< 3` → FULL_GPU | `agent.py:174` |
| rPPG cardiac band max | 4.0 Hz (240 BPM) | 2.5 Hz (150 BPM) | `thresholds.py:120` |
| rPPG hair occlusion | 35.0 | 0.25 | `thresholds.py:121` |

### Weight Mismatches (registry.py vs. thresholds.py)

| Tool | `registry.py` | `thresholds.py` |
|---|---|---|
| `run_dct` | 0.07 | 0.04 |
| `run_geometry` | 0.18 | 0.08 |
| `run_illumination` | 0.05 | 0.04 |
| `run_corneal` | 0.07 | 0.04 |
| `run_univfd` | 0.20 | 0.22 |
| `run_sbi` | 0.20 | 0.25 |
| `run_freqnet` | 0.09 | 0.10 |

### Security Notes

- No authentication or rate limiting on `/api/analyze`
- No filename sanitization (path traversal risk)
- No file cleanup after analysis (disk exhaustion risk)
- No upload collision handling (same filename overwrites)

### Code Quality Issues

| # | File:Line | Issue |
|---|---|---|
| 1 | `run_web.py:18` | `cpu_tools` list is dead code |
| 2 | `run_web.py:23` | Duplicate `JSONResponse` import |
| 3 | `config.py:36` | `clip_adapter_weights` references non-existent file |
| 4 | `thresholds.py:271` | Duplicate `RPPG_COHERENCE_THRESHOLD_HZ` definition |
| 5 | `llm.py:107,112` | `AgentEvent` imported inside function body |
| 6 | `memory.py` | Entire `MemorySystem` is unused in web pipeline |
| 7 | `registry.py:362` | Circuit breaker uses `time.time()` instead of `time.monotonic()` |

---

## Repository File Tree (Actual)

```
aegis-x/
├── run_web.py
├── setup.py
├── verify_tools.py
├── evaluate_pipeline.py
├── download_subset.py
├── pyproject.toml
├── requirements-main.txt
├── requirements-gpu.txt
├── .env.example
├── .gitignore
├── README.md
├── allfiles.md
├── questions.md
├── NOTES.md
├── DUAL_PATHWAY_PIPELINES_DESIGN.md
├── LICENSE
│
├── core/
│   ├── agent.py
│   ├── base_tool.py
│   ├── config.py
│   ├── data_types.py
│   ├── early_stopping.py
│   ├── exceptions.py
│   ├── forensic_summary.py
│   ├── llm.py
│   ├── memory.py
│   ├── subprocess_proxy.py
│   ├── subprocess_worker.py
│   └── tools/
│       ├── registry.py
│       ├── c2pa_tool.py
│       ├── dct_tool.py
│       ├── geometry_tool.py
│       ├── illumination_tool.py
│       ├── corneal_tool.py
│       ├── rppg_tool.py
│       ├── univfd_tool.py
│       ├── xception_tool.py
│       ├── sbi_tool.py
│       ├── freqnet_tool.py
│       └── freqnet/
│           ├── preprocessor.py
│           ├── fad_hook.py
│           └── calibration.py
│
├── utils/
│   ├── ensemble.py
│   ├── preprocessing.py
│   ├── thresholds.py
│   ├── vram_manager.py
│   ├── video.py
│   ├── image.py
│   ├── logger.py
│   └── ollama_client.py
│
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/                        ← gitignored
│   ├── clip-vit-large-patch14/
│   ├── univfd/probe.pth
│   ├── xception/xception_deepfake.pth
│   ├── sbi/efficientnet_b4.pth
│   └── freqnet/cnndetect_resnet50.pth
│
└── logs/
```

---

## Tool Manifest (Actual — from registry.py lines 78–89)

> **Note:** Registry weights sum to **1.12**, not 1.00. The `thresholds.py` constants (WEIGHT_*) reflect different values. The registry is the authoritative runtime source for `ToolSpec` metadata, but the ensemble scorer (`utils/ensemble.py`) uses `WEIGHT_MAP` from `thresholds.py`. This means early stopping and ensemble scoring use **different weight values** for the same tools.

| Tool name | Registry Weight | thresholds.py Weight | Category | Trust Tier | GPU? |
|---|---|---|---|---|---|
| `check_c2pa` | 0.05 | — | PROVENANCE | 1 | No |
| `run_dct` | 0.07 | 0.04 | FREQUENCY | 2 | No |
| `run_rppg` | 0.06 | 0.06 | BIOLOGICAL | 2 | No |
| `run_geometry` | 0.18 | 0.08 | GEOMETRIC | 3 | No |
| `run_illumination` | 0.05 | 0.04 | FREQUENCY | 1 | No |
| `run_corneal` | 0.07 | 0.04 | BIOLOGICAL | 2 | No |
| `run_univfd` | **0.20** | 0.22 | SEMANTIC | 3 | Yes (proxy) |
| `run_xception` | **0.15** | 0.15 | SEMANTIC | 2 | Yes (proxy) |
| `run_sbi` | **0.20** | 0.25 | GENERATIVE | 3 | Yes (proxy) |
| `run_freqnet` | 0.09 | 0.10 | FREQUENCY | 1 | Yes (proxy) |

> GPU tools are wrapped in `SubprocessToolProxy` — they never directly load PyTorch in `.venv_main`.

---

## Part 1 — Core Infrastructure

---

### `core/data_types.py` — ToolResult Contract

**Classes:** `ToolResult`

`ToolResult` is the universal output contract for every forensic tool. Uses `@dataclass(init=False)` with a custom `__init__` to resolve the `score` vs `fake_score` API split.

```
Fields:
  tool_name        str              Canonical tool name (e.g. "run_univfd")
  success          bool             True if inference completed without error
  fake_score       float            Primary field: P(fake) ∈ [0.0, 1.0]
  confidence       float            How reliable this score is ∈ [0.0, 1.0]
  details          Dict[str, Any]   Tool-specific metadata dict
  error            bool             True if an exception occurred
  error_msg        Optional[str]    Exception message string
  execution_time   float            Seconds elapsed
  evidence_summary str              Human-readable finding for LLM prompt

Property:
  score            → alias for fake_score (backward compat)
```

**Key behavior:** `__init__` accepts both `score=` and `fake_score=` kwargs; `fake_score` wins if both provided.

**What it cannot do:** No validation of score range — callers must clamp to [0.0, 1.0] themselves.

---

### `core/exceptions.py` — Exception Hierarchy

Defines custom exception classes for cross-process error propagation:

| Exception | Inherits | Use |
|---|---|---|
| `AegisBaseError` | `Exception` | Root of all Aegis-X exceptions |
| `ToolExecutionError` | `AegisBaseError` | Tool runtime failure |
| `VRAMError` | `AegisBaseError` | GPU memory management failure |
| `PreprocessingError` | `AegisBaseError` | Media processing failure |

---

### `core/config.py` — Configuration Hierarchy

All dataclasses. Reads from env vars via `os.getenv()` with defaults. Loaded once at import time via `load_dotenv()`.

| Dataclass | Fields | Purpose |
|---|---|---|
| `ModelPaths` | `phi3_model`, `univfd_backbone_dir`, `univfd_probe_path`, `xception_weights`, `sbi_weights`, `freqnet_weights`, `clip_adapter_weights` | Filesystem paths to model weight files |
| `AgentConfig` | `ollama_endpoint`, `ollama_model_name`, `llm_temperature=0.1`, `llm_max_tokens=1024`, `llm_context_window=4096` | Ollama LLM parameters |
| `EnsembleWeights` | One float per tool | Default weights (NOTE: uses `WEIGHT_UNIVFD=0.15` from thresholds; registry overrides at runtime) |
| `ThresholdConfig` | `real_threshold`, `fake_threshold`, `early_stop_confidence` | Decision thresholds |
| `PreprocessingConfig` | `face_crop_size=224`, `sbi_crop_size=380`, `max_video_frames=300`, `min_video_frames=90` | Preprocessing parameters |
| `XceptionConfig` | `confidence_base=0.40`, `confidence_multiplier=1.6` | Xception-specific scoring |
| `FreqNetFusionConfig` | `neural_weight=0.70`, `fad_weight=0.30` | FreqNet dual-stream fusion |
| `AegisConfig` | All of the above as fields | Master config — instantiate once per request |

**What it cannot do:** `EnsembleWeights` field values are overridden by `registry.py` at runtime — changing `config.weights.univfd` does **not** change what the ensemble uses.

---

### `core/base_tool.py` — Abstract Base Tool

**Class:** `BaseForensicTool(ABC)`

| Method | Abstract? | What it does |
|---|---|---|
| `tool_name` | ✅ property | Returns canonical name string |
| `setup()` | ✅ | Initialize weights/paths before first inference |
| `_run_inference(input_data)` | ✅ | Core algorithm — implemented by each tool |
| `execute(input_data)` | ❌ (concrete) | Safety wrapper: calls `_run_inference`, catches all exceptions, stamps `execution_time`. Returns ABSTAIN `ToolResult` on crash |

**Critical behavior of `execute()`:**
- On exception: returns `ToolResult(success=False, score=0.0, confidence=0.0, error=True)`
- Note: error result uses `score=0.0` (pushes REAL) — different from `_make_error_result` in agent which uses `score=0.5` (neutral). The agent's `_safe_execute_tool` never reaches this path because it wraps in `ThreadPoolExecutor` first.

---

### `core/agent.py` — ForensicAgent (v3.0)

The orchestrator. Generator-based — yields `AgentEvent` objects for SSE streaming.

**Module-level constants:**
```python
GPU_VRAM_REQUIREMENTS = {
    "run_freqnet": 0.4,   # GB — pre-flight VRAM check
    "run_univfd":  0.6,
    "run_xception":0.5,
    "run_sbi":     0.8,
}

FACE_GATE_THRESHOLDS = {
    "min_confidence":        0.60,   # tracking coverage ratio
    "min_face_area_ratio":   0.01,   # peak face area / frame area
    "min_frames_with_faces": 0.30,   # % frames with any face
    "min_face_pixel_area":   2500,   # NOT currently checked in code
}
```

**Class:** `AgentEvent`
```
Fields: event_type (str), tool_name (str|None), data (dict)
No methods — plain data container for SSE.
```

**Class:** `ForensicAgent`

| Method | What it does |
|---|---|
| `__init__(config)` | Creates `ToolRegistry` singleton, `EnsembleAggregator`, `EarlyStoppingController` (thresholds `(0.5, 0.5)` — ESC not used for primary gating in v3) |
| `_make_error_result(tool_name, error_msg, start_time)` | **DRY factory** — returns `ToolResult(success=False, score=0.5, confidence=0.0, error=True)`. Used by all 4 error paths |
| `_safe_execute_tool(tool_name, input_data, timeout=30)` | CPU tool runner. Wraps `tool.execute()` in `ThreadPoolExecutor(max_workers=1)`, calls `future.result(timeout=30)`. On `FuturesTimeoutError` or `Exception` → `_make_error_result()` |
| `analyze(preprocess_result, media_path)` | **Main generator pipeline** — see full flow below |

**`analyze()` — Full Flow:**

```
1. Build input_data dict from preprocess_result
2. FACE GATE (4 checks):
   has_face AND max_confidence ≥ 0.60
   AND max_face_area_ratio ≥ 0.01
   AND frames_with_faces_pct ≥ 0.30
   → pass_face_gate: bool
   yield PIPELINE_SELECTED event

3. SEGMENT A — CPU Phase:
   Base tools: [check_c2pa, run_dct]
   If pass_face_gate:
     Add run_rppg (video only, not insufficient_temporal_data)
     Add run_geometry (unless MOTION_BLUR/OCCLUSION flag)
     Add run_illumination (unless MOTION_BLUR/OCCLUSION/LOW_LIGHT)
     Add run_corneal (unless MOTION_BLUR/OCCLUSION/FACE_TOO_SMALL/LOW_LIGHT)
   For each: _safe_execute_tool(timeout=30) → ensemble.add_result()
   C2PA short-circuit: if c2pa_verified → return REAL immediately

4. SEGMENT B — CPU→GPU Gate:
   decisive_results = [r where |score - 0.5| > 0.15]  ← code uses 0.15 (not 0.05)
   If len < 3 → FULL_GPU                               ← code uses 3 (not 2)
   Else:
     direction_i = (score_i - 0.5) × 2
     agg_direction = Σ direction_i × confidence_i × normalized_weight_i
     agg_conf = |agg_direction|
     unison = all decisive_results agree on direction
     domains from {bio(rPPG), phys(geometry), freq(dct), auth(c2pa), bio_corn(corneal), illum(illumination)}
     unison_agreement = unison AND len(domains) ≥ 2
     HALT if agg_conf > 0.93 AND unison_agreement
     MINIMAL_GPU if agg_conf ≥ 0.80
     FULL_GPU otherwise
   yield GATE_DECISION event

5. SEGMENT C — GPU Phase (if gate != HALT):
   pass_face_gate: [freqnet, univfd, xception, sbi]
   no face:        [freqnet, univfd, xception]
   MINIMAL_GPU:    [univfd]
   For each:
     make_loader(tool) → lambda: tool   (closure-safe factory)
     make_inference(data) → lambda t: t.execute(data)
     ThreadPoolExecutor → run_with_vram_cleanup(timeout=60)
     FuturesTimeoutError: _make_error_result("Timeout after 60s")
     Exception: _make_error_result(str(e))
     Both error paths: torch.cuda.empty_cache()

6. DEGRADED check:
   is_degraded = errors/total > 0.50

7. Ensemble + LLM:
   final_score = ensemble.get_final_score()
   verdict_str = ensemble.get_verdict()
   explanation = await generate_verdict(...)
   yield VERDICT {verdict, score, explanation, degraded}
   return {verdict, score, explanation, degraded}
```

**What it cannot do:**
- Does not parallelize GPU tools (sequential by design for VRAM safety)
- ESC (`self.esc`) is instantiated but not called in the primary flow — the inline directional gate replaces it
- Does not retry failed tools

---

### `core/early_stopping.py` — Evidential Subjective Logic Controller

**Classes:** `StopReason` (Enum), `StopDecision` (dataclass), `EarlyStoppingController`

`EarlyStoppingController` is instantiated by `ForensicAgent` but **not used in the primary v3.0 gate flow**. The inline directional gate in `analyze()` Segment B supersedes it. It remains available for future re-integration.

**`StopReason` values:**
```
CONTINUE_AMBIGUOUS             — default continue
CONTINUE_ADVERSARIAL_CONFLICT  — conflict ratio > threshold, force continue
CONTINUE_SECURITY_REQUIRED     — high confidence but no high-trust tool yet
HALT_C2PA_HARDWARE_SIGNED      — cryptographic verification
HALT_LOCKED_FAKE               — mathematical bound guarantees FAKE
HALT_LOCKED_REAL               — DISABLED in v3.0 (commented out)
```

**`EarlyStoppingController.evaluate()` — 9-step algorithm:**

1. C2PA hardware lock → immediate HALT
2. Empty scores → CONTINUE
3. Validate tool names against registry
4. Compute `weights_run`, `weights_pending` via `get_viable_pending_tools()`
5. Weighted mean: `current_score = Σ(score × weight) / weights_run`
6. Evidential Subjective Logic:
   - `e_fake = Σ weight × max(0, score−0.5) × 2`
   - `e_real = Σ weight × max(0, 0.5−score) × 2`
   - `conflict_ratio = min(e_fake, e_real) / max(e_fake, e_real)`
   - If `conflict_ratio > 0.35` → CONTINUE_ADVERSARIAL_CONFLICT
7. Mathematical bounds: `max_possible`, `min_possible` given pending tools
8. HALT_LOCKED_REAL — **disabled** (comment in code explains: classical tools score ~0.0 on generative fakes, creating false mathematical ceiling)
9. HALT_LOCKED_FAKE if `min_possible > real_threshold AND current_score > fake_threshold`

**What it cannot do:** Cannot detect directional disagreement between tools (e.g. rPPG says REAL, geometry says FAKE at equal magnitudes). The v3.0 inline gate handles this via signed directional scoring.

---

### `core/forensic_summary.py` — LLM Prompt Builder

**Functions:**

| Function | Signature | Returns |
|---|---|---|
| `build_phi3_prompt` | `(ensemble_score, tool_results, verdict) → str` | Structured plain-text prompt for Phi-3 Mini |
| `generate_verdict` | `async (ensemble_score, tool_results, verdict) → str` | Calls `stream_completion`, concatenates token stream, returns full explanation |

**`build_phi3_prompt()` structure:**
```
Block 1: CRITICAL score interpretation (all scores = REAL probability)
Block 2: Header (ensemble_score %, verdict)
Block 3: SECONDARY HELPERS (C2PA, rPPG, DCT, Geometry, Illumination, Corneal)
Block 4: PRIMARY AI DETECTORS (UnivFD, XceptionNet, SBI, FreqNet)
Block 5: REASONING RULES (6 rules — trust primaries over secondaries)
Block 6: OUTPUT FORMAT (plain paragraph, no JSON/markdown)
```

**`_interpret()` helper** (inner function):
- Converts `fake_score → real_prob = 1.0 - score`
- `real_prob ≥ 0.70` → CLEAR
- `real_prob ≥ 0.45` → INCONCLUSIVE
- `else` → SUSPICIOUS

**What it cannot do:**
- LLM never sees raw pixels — only structured text (enforced by design)
- `generate_verdict` is `async` — callers must `await` or use `yield from` (agent.py does `yield from generate_verdict(...)` — this is a bug risk since `generate_verdict` is a coroutine, not a generator. Works in practice because `generate_verdict` is awaited via `stream_completion`).

---

### `core/llm.py` — Ollama HTTP Client Bridge

Thin wrapper around the Ollama REST API (`/api/generate`).

| Function/Class | What it does |
|---|---|
| `stream_completion(prompt, temperature, max_tokens)` | Async generator — yields string tokens from Ollama streaming response |
| `generate_verdict(ensemble_score, tool_results, verdict)` | **Sync generator** — bridges async OllamaClient via `threading.Thread` + `queue.Queue`. Yields `AgentEvent("llm_stream")` tokens, returns full explanation string |
| `OllamaClient` | Configurable async client with endpoint, model name, retry logic, health checks |

**⚠️ Critical Bug at line 100:** `logger.error("LLM generation timed out")` references `logger` which is **never imported or defined** in this module. When the LLM times out (queue.Empty after 300s), this raises `NameError: name 'logger' is not defined`. The `from utils.logger import setup_logger` import is missing.

Uses `httpx` for async HTTP. Temperature defaults to 0.1 (deterministic forensic output). `keep_alive=0` is set intentionally for low-VRAM systems (see `ollama_client.py:88`).

**Architecture note:** `generate_verdict()` creates a new thread per call with `threading.Thread(target=async_runner)`, then blocks on `t.join()`. This is effectively synchronous despite the async internals.

---

### `core/memory.py` — Case Memory System

SQLite-backed persistent case storage. Not used in the primary analysis pipeline — available for case history retrieval in the web UI.

| Method | What it does |
|---|---|
| `store_case(media_hash, result_dict)` | Persists verdict + all tool scores by content hash |
| `get_case(media_hash)` | Retrieves cached result for previously analyzed media |
| `list_recent(n)` | Returns last N analyzed cases |

**What it cannot do:** Does not deduplicate compressed/resized versions of the same image — only exact hash matches.

---

### `core/subprocess_proxy.py` — `.venv_main` ↔ `.venv_gpu` Bridge

**Class:** `SubprocessToolProxy(BaseForensicTool)`

Wraps any GPU tool to run in the `.venv_gpu` process. GPU tools in the registry are never directly instantiated — they are replaced with `SubprocessToolProxy` instances.

| Method | What it does |
|---|---|
| `__init__(tool_name)` | Stores tool name; no subprocess yet |
| `execute(input_data)` | Serializes `input_data` (pickle), spawns `.venv_gpu/bin/python subprocess_worker.py`, sends to stdin, reads `ToolResult` from stdout |
| `tool_name` | Returns the wrapped tool name |

**What it cannot do:** Cannot stream intermediate results from GPU worker — only final `ToolResult`.

---

### `core/subprocess_worker.py` — GPU Worker Process

Runs inside `.venv_gpu`. Reads pickled `input_data` from stdin, dynamically imports the real tool class, calls `execute()`, writes pickled `ToolResult` to stdout.

**What it cannot do:** Has no timeout enforcement of its own — timeout is enforced by `ThreadPoolExecutor` in `agent.py` Segment C.

---

## Part 2 — Tool Registry

---

### `core/tools/registry.py` — ToolRegistry

**Enums/Dataclasses:** `ToolCategory`, `ToolSpec`

**`ToolCategory`** values: `FREQUENCY`, `SEMANTIC`, `GEOMETRIC`, `PROVENANCE`, `GENERATIVE`, `BIOLOGICAL`

**`ToolSpec`** (frozen dataclass):
```
Fields: name (str), weight (float > 0), category (ToolCategory), trust_tier (int ∈ {1,2,3})
trust_tier 3 = high accuracy (geometry, univfd, sbi)
trust_tier 1 = lightweight, artifact-prone (c2pa, illumination, freqnet)
```

**`ToolRegistry`** — singleton via `get_registry()` (double-checked locking with `threading.Lock`):

| Method | What it does |
|---|---|
| `__init__()` | Builds metadata registry from `_TOOL_MANIFEST`; calls `_register_all()` |
| `_register_all()` | Iterates manifest. GPU tools → `SubprocessToolProxy`. CPU tools → direct import → `instance.setup()`. Each tool isolated — one failure doesn't block others |
| `get_tool(name)` | Returns `BaseForensicTool` instance (or proxy) by name |
| `get_tool_spec(name)` | Returns `ToolSpec` with weight/category/tier |
| `get_high_trust_tools()` | Returns names of tier-3 tools |
| `get_viable_pending_tools(completed)` | Returns registered tools not in `completed` list (for ESC) |
| `execute_tool(name, input_data)` | Direct execution with OOM retry (not used by agent in v3.0 — agent calls `get_tool()` directly) |
| `get_health_report()` | Dict with active/failed counts + per-tool execution stats |
| `shutdown()` | Calls `cleanup()` on each tool, clears VRAM |

**GPU tool registration:** All 4 GPU tools are wrapped as `SubprocessToolProxy` instances — they never load PyTorch into `.venv_main`.

**`get_registry()` / `reset_registry()`**: Module-level singleton functions. `reset_registry()` for testing only.

---

## Part 3 — CPU Forensic Tools

---

### `core/tools/c2pa_tool.py` — C2PA Provenance Verificaton

**Class:** `C2PATool(BaseForensicTool)` | Weight: 0.05 | Tier: 1 | GPU: No

**What it does:** Reads C2PA Content Credentials from the input file using the `c2pa-python` library. Extracts signer identity and timestamp from the active manifest's `signature_info`.

| Method | What it does |
|---|---|
| `setup()` | Tries `import c2pa`; sets `_c2pa_available` flag |
| `_run_inference(input_data)` | Reads `media_path`; tries `c2pa.read_file()` first, falls back to `c2pa.Reader()` API for library version compat |
| `_no_c2pa_result(start_time)` | Returns `success=True, confidence=0.0` abstention when no C2PA data present |

**Score logic:**
- `score=0.0, confidence=1.0` → C2PA verified (cryptographic chain intact) → agent triggers short-circuit to REAL
- `score=0.0, confidence=0.0` → No C2PA data (abstention — unsigned media is common)
- `c2pa_verified=True` in `details` is what agent reads for short-circuit, NOT the score

**What it cannot do:**
- Cannot validate *what* the signer claims — only that the signature chain is valid
- Cannot detect if C2PA data was stripped (image re-saved without metadata)
- Score `0.0` for both verified AND unverified — only `c2pa_verified` flag differentiates

---

### `core/tools/dct_tool.py` — JPEG Double-Quantization Detector

**Class:** `DCTTool(BaseForensicTool)` | Weight: 0.07 | Tier: 2 | GPU: No

**What it does:** Detects JPEG double-quantization artifacts — the statistical fingerprint left when a JPEG is re-saved (compressed twice). Uses DCT-II transform on 8×8 blocks, computes AC coefficient histogram autocorrelation. Secondary peak in autocorrelation = double-quantization.

| Method | What it does |
|---|---|
| `setup()` | No-op |
| `_coerce_to_uint8(frame)` | Handles float [0,1] and float [0,255] inputs safely |
| `_to_gray(crop)` | BT.601 RGB→gray (0.299R + 0.587G + 0.114B) — RGB order assumed |
| `_compute_video_hash(frames)` | MD5 of top-left 100×100 of first frame for grid cache key |
| `_find_optimal_grid(gray, hash)` | Tests all 64 (dy,dx) ∈ [0,7]² alignments; picks the one maximizing autocorrelation secondary peak ratio. Cached per video |
| `_compute_peak_ratio(gray, dy, dx)` | Computes `secondary_peak / primary_peak` of AC coefficient autocorrelation |
| `_score_from_ratio(ratio)` | `clamp((ratio - DCT_RATIO_THRESHOLD) / DCT_RATIO_SCALE, 0, 1)` where threshold=0.75, scale=0.15 |
| `_run_inference(...)` | Iterates tracked faces; uses `face_crop_224` only (no frame fallback); averages `peak_ratio` across faces |

**Output details keys:** `grid_artifacts` (bool), `peak_ratio` (float), `faces_analyzed` (int), `grid_alignment` (dy,dx tuple)

**What it cannot do:**
- Blind to PNG deepfakes (no JPEG compression artifacts to detect)
- Blind to lossless copies of JPEG deepfakes
- High-quality JPEG (Q95+) resaves may not show detectable secondary peak

---

### `core/tools/rppg_tool.py` — Remote Photoplethysmography (v3.0)

**Class:** `RPPGTool(BaseForensicTool)` | Weight: 0.06 (ensemble) / 0.35 (gate) | Tier: 2 | GPU: No | Video only

**What it does:** Extracts cardiac pulse signal from facial color changes using the **POS (Plane-Orthogonal-to-Skin)** algorithm on 3 ROIs (forehead, left cheek, right cheek). Analyzes via FFT in the cardiac frequency band (0.7–2.5 Hz). Returns one of 9 liveness labels.

**Key thresholds from `thresholds.py`:**
```
RPPG_MIN_FRAMES = 90              # Skip if < 90 frames extracted
RPPG_HAIR_OCCLUSION_VARIANCE = 0.25  # ⚠️ BUG: Should be ~35.0 per code comments. Currently 0.25 causes false positives on nearly every frame.
RPPG_CARDIAC_BAND_MIN_HZ = 0.7   # 42 BPM
RPPG_CARDIAC_BAND_MAX_HZ = 2.5   # 150 BPM (⚠️ README documents 4.0 Hz / 240 BPM — code uses 2.5)
RPPG_COHERENCE_THRESHOLD_HZ = 0.5 # Max inter-ROI peak difference for PULSE_PRESENT
RPPG_SNR_THRESHOLD = 3.0         # Min spectral_concentration for "good" ROI
```

> ⚠️ **RPPG_HAIR_OCCLUSION_VARIANCE = 0.25** in thresholds.py but `_evaluate_liveness` hardcodes the comment "variance > 35.0 indicates hair texture". The `_check_hair_occlusion` method uses `RPPG_HAIR_OCCLUSION_VARIANCE` which is 0.25 — nearly every non-black ROI will trigger this. **Confirmed bug: threshold is ~100x too low.**

| Method | What it does |
|---|---|
| `setup()` | Sets `_debug = False` |
| `_extract_roi(frame, bbox, relative_box)` | Clips relative ROI box to frame bounds; returns frame crop |
| `_get_facial_rois(landmarks)` | Returns 3 ROI relative boxes; refines using actual 478-pt MediaPipe landmarks if available |
| `_check_hair_occlusion(roi)` | Laplacian variance of ROI — returns `(is_occluded, variance)` |
| `_extract_pos_signal(frames, trajectory, roi)` | POS algorithm: normalize RGB means → multiply by POS weights → standardize H signal |
| `_calculate_signal_metrics(signal, fps)` | FFT → PSD → cardiac band peak → spectral_concentration = peak_power/median_power → SNR in dB |
| `_evaluate_liveness(h_f, h_l, h_r, stds, hair)` | Decision tree → one of 9 labels (see table below) |
| `_lightweight_face_check(frames)` | MediaPipe FaceDetection on 5 sampled frames — backup when tracking failed |
| `_run_inference(input_data)` | Main entry. Returns SKIPPED (image), ABSTAIN (< min frames). For each face: **v3.0 fix** — `face_window=(0,0)` → ABSTAIN+continue; else slice frames and run POS |

**Liveness label table:**

| Label | score | confidence | Meaning |
|---|---|---|---|
| `PULSE_PRESENT` | 0.0 | 0.70–0.95 | ≥2 ROIs coherent at same cardiac freq |
| `ABSTAIN` | 0.0 | 0.0 | face_window failed / skipped |
| `SKIPPED` | 0.0 | 0.0 | Static image input |
| `NO_PULSE` | 1.0 | 0.90 | All regions flat OR no cardiac peak |
| `SYNTHETIC_FLATLINE` | 1.0 | 0.85 | < 2 ROIs have temporal variance |
| `WEAK_PULSE_FAILED` | 1.0 | 0.70 | Only 1 ROI passes quality threshold |
| `INCOHERENT` | 1.0 | 0.65–0.90 | ROI peaks don't synchronize |
| `AMBIGUOUS` | 0.0 | 0.0 | Hair occlusion (texture variance check) |
| `TRACKING_FAILED` | 1.0 | 0.85 | All ROI signals returned None (non-hair) |

**What it cannot do:**
- Cannot detect pulse in images (static media)
- Cannot detect pulse in < 90 frames (~3 sec at 30fps)
- rPPG weight in ensemble (0.06) ≠ rPPG weight in CPU→GPU gate (0.35) — two separate systems

---

### `core/tools/geometry_tool.py` — Facial Proportion Analyzer

**Class:** `GeometryTool(BaseForensicTool)` | Weight: 0.18 | Tier: 3 | GPU: No

**What it does:** Validates 6 anthropometric facial ratios using MediaPipe 468-point landmarks. Detects anatomically impossible proportions common in facial warping/deepfake synthesis.

**Ratios checked:**
```
IPD ratio       (inter-pupillary distance / face width): [0.42, 0.52]
Philtrum ratio  (nose-to-lip / face height):             [0.10, 0.15]
Eye asymmetry   (|left_eye_h - right_eye_h| / face_h):  max 0.05
Nose width      (nose_width / IPD):                      [0.55, 0.70]
Mouth width     (mouth_width / IPD):                     [0.85, 1.05]
Vertical thirds (deviation from equal thirds):           max 0.15
```

**Score:** `violations / total_checks`. Each passing check = 0.0 contribution; each violation = 1/N.

**What it cannot do:**
- Skips faces with yaw > 18% of face width (side-facing)
- Cannot detect subtle warping that stays within anthropometric bounds
- Score of 0.0 is ambiguous — can mean "perfectly real" or "face too rotated to check"

---

### `core/tools/illumination_tool.py` — Lighting Consistency Analyzer

**Class:** `IlluminationTool(BaseForensicTool)` | Weight: 0.05 | Tier: 1 | GPU: No

**What it does:** Detects lighting inconsistencies using gradient-based directional analysis. Computes gradient vectors across facial regions; inconsistent lighting direction = splice composite.

**What it cannot do:**
- Abstains (confidence=0.0) when gradient magnitude is too diffuse (< ILLUMINATION_DIFFUSE_THRESHOLD=0.05)
- Cannot detect subtle global lighting adjustments
- High false-positive rate on complex natural lighting setups

---

### `core/tools/corneal_tool.py` — Corneal Catchlight Analyzer

**Class:** `CornealTool(BaseForensicTool)` | Weight: 0.07 | Tier: 2 | GPU: No

**What it does:** Detects catchlight (specular reflection) position and consistency in both eyes using a 15×15 pixel sampling box. Divergent or absent catchlights indicate composite lighting mismatches.

**Thresholds:** `CORNEAL_BOX_SIZE=15`, `CORNEAL_MAX_DIVERGENCE=0.5`, `CORNEAL_REFLECTION_THRESHOLD=0.75`

**What it cannot do:**
- Abstains when face crop too small (< FACE_TOO_SMALL flag triggers skip in agent)
- Cannot detect indirect/diffuse lighting setups (legitimate lack of catchlights scores as suspicious)

---

## Part 4 — GPU Forensic Tools

---

### `core/tools/univfd_tool.py` — UnivFD CLIP-Based Detector

**Class:** `UnivFDTool(BaseForensicTool)` | Registry Weight: **0.20** | Tier: 3 | GPU: Yes (via proxy)

**Architecture:**
```
CLIPVisionModelWithProjection (frozen, FP16) → (1, 768)
L2 normalize → (1, 768)
_LinearProbe: Linear(768, 1) (FP32)
Sigmoid → scalar [0.0, 1.0]
```

**`_LinearProbe`** class: `nn.Module` wrapping `nn.Linear(768, 1)`. Loaded from `models/univfd/probe.pth` with 3-format key detection (PyTorch native / sklearn dict / raw tensor).

| Method | What it does |
|---|---|
| `setup()` | Checks backbone dir + probe path; sets handles to None |
| `_load_model()` | CLIP backbone FP16 + probe FP32, both to `cuda:0` |
| `unload()` | `model.cpu() → synchronize() → del → gc.collect() → empty_cache()` |
| `_crop_to_tensor(crop, device)` | `CLIPImageProcessor` for 224×224 (CLIP normalization stats, not ImageNet) |
| `_score_single_crop(crop, device)` | FP16 → FP32 cast → L2 norm → probe → sigmoid |
| `_run_inference(input_data)` | TTA: original + H-flip, max-pool; worst-face policy |

**Pre-trained weights:**
- Backbone: `models/clip-vit-large-patch14/` (~890 MB actual, not ~3.5 GB)
- Probe: `models/univfd/probe.pth` (~4 KB)

**VRAM pre-flight check:** `GPU_VRAM_REQUIREMENTS["run_univfd"] = 0.6 GB` (minimum free VRAM required). Actual peak VRAM ~1.8 GB when loaded.

**Backward-compat shim:** `result.details["siglip_score"]` mirrors `result.fake_score` for ensemble code referencing old key.

**What it cannot do:**
- Cannot detect face-swap/reenactment (no blend boundary signal) — XceptionNet covers this
- Adversarial examples crafted against CLIP-ViT can fool the probe
- CPU-only machines: abstains

---

### `core/tools/xception_tool.py` — XceptionNet Face-Swap Detector

**Class:** `XceptionTool(BaseForensicTool)` | Registry Weight: **0.15** | Tier: 2 | GPU: Yes (via proxy)

**Architecture:** `timm.create_model('xception', num_classes=2)` — Xception pretrained on FaceForensics++ (Rossler et al. ICCV 2019). Output = `softmax[:, 1]` = P(fake).

| Method | What it does |
|---|---|
| `setup()` | Checks `models/xception/xception_deepfake.pth` |
| `_load_model()` | `timm.create_model` + checkpoint loaded with `_remap_keys()` |
| `_remap_keys(state_dict)` | Strips `module.`/`model.` prefixes; renames `last_linear.*` → `fc.*` for HongguLiu→timm compat |
| `unload()` | CPU offload → synchronize → del → gc → empty_cache |
| `_score_crop(crop, device)` | Resize to 299×299 (Lanczos4); ImageNet normalize; FP16 forward; softmax[:, 1] |
| `_run_inference(input_data)` | TTA (original + H-flip, max-pool); worst-face policy |

**VRAM:** `GPU_VRAM_REQUIREMENTS["run_xception"] = 0.5 GB` pre-flight check. Actual ~350 MB loaded (FP16).

**What it cannot do:**
- Not trained on generative AI (Midjourney/DALL-E/SD) — UnivFD covers this
- Very high-quality professional face swaps with feathered masks may evade detection

---

### `core/tools/sbi_tool.py` — Self-Blended Images Blend Detector

**Class:** `SBITool(BaseForensicTool)` | Registry Weight: **0.20** | Tier: 3 | GPU: Yes (via proxy)

**Architecture:** EfficientNet-B4 (`efficientnet_b4`, 1792 feature channels) + custom head: `Sequential(Dropout(0.4), Linear(1792, 1))`. Input: 380×380 dual-scale crops (1.15× and 1.25×). GradCAM with MediaPipe region mapping.

**Status:** Modified in v3.0 — threshold imports restructured, skip gate logic fixed.

| Method | What it does |
|---|---|
| `_load_model()` | EfficientNet-B4 with custom classifier head; loads from `models/sbi/efficientnet_b4.pth`; `strict=False` load |
| `_prepare_crop_and_landmarks(face, lm, scale)` | BORDER_CONSTANT padding → 380×380 Lanczos4 → affine-correct landmark transform |
| `_compute_gradcam(model, tensor)` | Pass 2: hooks `model.features[-1]`, runs forward+backward, computes CAM, resizes to 380×380 |
| `_map_regions(cam, landmarks)` | Maps CAM peaks to 5 named landmark regions; 5% border clipping (px 19–360) to avoid BORDER_CONSTANT false positives |
| `_run_inference(input_data)` | Skip if `visual_score > SBI_SKIP_CLIP_THRESHOLD (0.70)` (fully synthetic — no blend). Two-pass: Pass 1 no-grad score; Pass 2 conditional GradCAM if score > SBI_FAKE_THRESHOLD (0.50) |

**VRAM:** `GPU_VRAM_REQUIREMENTS["run_sbi"] = 0.8 GB` — highest of all GPU tools. Actual peak ~700 MB (FP32 EfficientNet + GradCAM gradients).

**Score finalization logic:**
```python
final_score = best_score if (boundary_detected 
    OR (best_boundary_region == "diffuse" AND best_score > SBI_FAKE_THRESHOLD)) 
    else 0.0
```

**SBI named regions:** `jaw`, `hairline`, `cheek_l`, `cheek_r`, `nose_bridge`

**What it cannot do:**
- Blind below `SBI_FAKE_THRESHOLD (0.50)` — by design avoids false positives
- Fully synthetic images (no blend boundary) → skipped when UnivFD score is high
- GradCAM region mapping only works with valid MediaPipe landmarks

---

### `core/tools/freqnet_tool.py` — CNNDetect + FADHook Frequency Detector

**Class:** `FreqNetTool(BaseForensicTool)` | Weight: 0.09 | Tier: 1 | GPU: Yes (via proxy)

**Architecture:** Dual-stream fusion:
1. **Neural stream:** `_CNNDetect` (ResNet-50 backbone + `Linear(2048, 1)`) pretrained on ProGAN
2. **Statistical stream:** `DCTPreprocessor` + `FADHook` — pure-math BT.709 DCT-II, zero weights

Final score: `0.70 × neural_score + 0.30 × fad_score`
Fallback (no weights): `1.0 × fad_score`

**`_CNNDetect(nn.Module)`:**
```python
self.features   = nn.Sequential(*list(resnet50.children())[:-1])  # → (B, 2048, 1, 1)
self.classifier = nn.Linear(2048, 1, bias=True)
forward: features(x).flatten(1) → classifier → (B, 1) logit
```

| Method | What it does |
|---|---|
| `setup()` | Loads `CalibrationManager`; checks for CNNDetect weights |
| `_load_model()` | Builds `_CNNDetect`; loads `cnndetect_resnet50.pth` with `_remap_cnndetect_keys()` |
| `_remap_cnndetect_keys(ckpt)` | Strips `module.`/`model.` prefixes; renames `fc.*` → `classifier.*` |
| `unload()` | CPU offload → synchronize → del → gc → empty_cache |
| `_run_fad_analysis(crop, device)` | `DCTPreprocessor._dct_conv` hook → `FADHook.analyze()` → band energy excess → fad_score |
| `_run_inference(input_data)` | 10% crop expansion; neural stream (FP16 ResNet-50); FAD stream; fuse 70/30; worst-face |

**VRAM:** `GPU_VRAM_REQUIREMENTS["run_freqnet"] = 0.4 GB` — lowest of GPU tools. Actual ~270 MB loaded.

**What it cannot do:**
- Post-processed deepfakes (Gaussian blur, median filter destroy frequency fingerprint)
- Frequency-domain adversarial attacks have published bypass methods
- Unusually sharpened real camera images may false-positive

---

## Part 5 — FreqNet Subpackage: `core/tools/freqnet/`

---

### `preprocessor.py` — DCTPreprocessor + SpatialPreprocessor

**`DCTPreprocessor(nn.Module)`:**
- BT.709 luma extraction → DCT-II conv2d (64 frozen cosine basis function kernels)
- `log1p` compression → output shape `(B, 64, 28, 28)`
- `_dct_conv` is the hooked layer for FADHook

**`SpatialPreprocessor(nn.Module)`:**
- ImageNet normalization with registered buffers (mean/std as non-trainable params)
- Used for CNNDetect input preprocessing

---

### `fad_hook.py` — FADHook (Frequency Artifact Detection)

**`FADHook`:** Forward hook capturing `(B, 64, 28, 28)` DCT tensors from `DCTPreprocessor`.

**`BandAnalysis` dataclass:**
```
base_energy_ratio   float  # Low-freq energy fraction
mid_energy_ratio    float  # Mid-freq energy fraction
high_energy_ratio   float  # High-freq energy fraction
mid_z_score         float  # Z-score vs calibrated baseline
high_z_score        float  # Z-score vs calibrated baseline
```

Band division: JPEG zigzag ordering — base (i+j ≤ 2), mid (3 ≤ i+j ≤ 5), high (i+j > 5).

**`fad_score` computation:** `max(0, mid_excess) × FREQNET_FAD_MID_MULTIPLIER + max(0, high_excess) × FREQNET_FAD_HIGH_MULTIPLIER` where excess = observed − expected from CalibrationManager.

---

### `calibration.py` — CalibrationManager

Loads `freqnet_fad_baseline.pt` for Z-score normalization. Falls back to hardcoded natural-image statistics if file absent: `base=0.70, mid=0.25, high=0.05`.

---

## Part 6 — Utils

---

### `utils/thresholds.py` — Single Source of Truth for All Numeric Constants

**DO NOT** hardcode any numeric value elsewhere in the codebase. All tools import from here.

Key groups:
- Hardware/VRAM: `VRAM_MIN_FOR_GPU=3.5`, `VRAM_RESERVED_BUFFER_GB=1.0`
- Ensemble weights (used by ensemble.py — different from registry.py weights)
- Decision thresholds: `ENSEMBLE_REAL_THRESHOLD=0.50`, `ENSEMBLE_FAKE_THRESHOLD=0.50`
- Per-tool: DCT, Geometry, Illumination, Corneal, rPPG, SBI, FreqNet, UnivFD, Xception, C2PA
- Agent: `EARLY_STOP_CONFIDENCE=0.75`, `SUSPICION_OVERRIDE_THRESHOLD=0.70`

**⚠️ Critical bugs in this file:**
- Line 120: `RPPG_CARDIAC_BAND_MAX_HZ = 2.5` (README documents 4.0 Hz)
- Line 121: `RPPG_HAIR_OCCLUSION_VARIANCE = 0.25` (code comments say 35.0 — ~100x too low)
- Line 271: `RPPG_COHERENCE_THRESHOLD_HZ = 0.5` duplicate of line 123 (harmless but redundant)

**`ThresholdConfig` (dataclass, frozen):** `real_threshold=0.15`, `fake_threshold=0.85` — used by ESC, not ensemble.

---

### `utils/ensemble.py` — EnsembleAggregator (v4.0)

**Important:** Uses `WEIGHT_MAP` from `thresholds.py` — **NOT** the registry weights. This is the weight discrepancy source.

**`WEIGHT_MAP`** (from thresholds): univfd=0.22, xception=0.15, sbi=0.25, freqnet=0.10, rppg=0.06, dct=0.04, geometry=0.08, illumination=0.04, corneal=0.04 — different from registry (0.20/0.15/0.20/0.09/0.06/0.07/0.18/0.05/0.07).

**Classes:** `EnsembleAggregator`

| Method | What it does |
|---|---|
| `add_result(result)` | Appends `ToolResult` to `tool_results` dict |
| `get_final_score()` | Calls `calculate_ensemble_score()` |
| `get_verdict()` | "FAKE" if score ≤ ENSEMBLE_REAL_THRESHOLD (0.50), else "REAL" |
| `calculate_ensemble_score(results)` | Full 4-step scoring pipeline |

**`calculate_ensemble_score()` — 4-step pipeline:**

```
Step 1: Extract context
  dct_peak_ratio from "run_dct" result.details
  compression_flag = dct.details["grid_artifacts"] if present

Step 2: C2PA override check
  If c2pa_verified AND no visual contradiction (GPU specialist < C2PA_VISUAL_CONTRADICTION_THRESHOLD)
  → return 1.0 (REAL) immediately

Step 3: Route each tool via _route()
  rPPG:    PULSE_PRESENT → RPPG_NO_PULSE_IMPLIED_PROB (0.85) fake
           ABSTAIN/AMBIGUOUS → skip
           NO_PULSE → implied_prob=0.85 → contribution
  SBI:     below SBI_BLIND_SPOT_THRESHOLD (0.50) → skip (no vote)
           mid-band: base weight + clip_multiplier × univfd_context
           above threshold: direct contribution
  FreqNet: below FREQNET_BLIND_SPOT_THRESHOLD (0.45) → skip
           compression discount applied if DCT flag set
  Others:  score × weight direct

Step 4: Three-Pronged Anomaly Shield
  PRONG 1 — Suspicion Overdrive:
    max_gpu_prob = max(GPU specialists' implied fake probs)
    gpu_spread = max_gpu_prob - min_gpu_prob
    if max_gpu_prob > SUSPICION_OVERRIDE_THRESHOLD (0.70):
      if gpu_spread > 0.30 AND len(gpu_specialists) >= 2:
        fake_score = weighted_average  # conflict → fallback
      else:
        fake_score = max_gpu_prob      # hard max-pool override

  PRONG 2 — Borderline Consensus:
    borderline = [p for p in gpu_probs if 0.35 <= p <= 0.55]
    if len(borderline) >= 2:
      consensus_anchor = mean(borderline) * 1.25

  PRONG 3 — GPU Coverage Degradation:
    for each abstained GPU specialist:
      gpu_degradation_boost *= (1.0 + 0.10)

  Final: fake_score = max(base_ensemble, anomaly_anchor, consensus_anchor) * degradation_boost

ensemble_score = 1.0 - fake_score
```

**What it cannot do:**
- Does not use the registry weights — uses `WEIGHT_MAP` from thresholds.py
- EMA smoothing constants exist but temporal smoothing is not applied per-call in current implementation

---

### `utils/vram_manager.py` — VRAM Lifecycle Manager

Hardware priority: TPU → CUDA → MPS → CPU

**`get_device()`:** Auto-detects best accelerator. TPU detection requires `torch_xla` and real TPU backing (not XLA-CPU fallback).

**Module functions:**

| Function | What it does |
|---|---|
| `get_device()` | Auto-detect TPU/CUDA/MPS/CPU |
| `log_vram_status(tag, device_id)` | Logs free/used/total VRAM at DEBUG level |
| `_get_available_vram_gb()` | Returns total VRAM in GB |
| `run_with_vram_cleanup(loader_fn, inference_fn, model_name, required_vram_gb)` | Main entry point used by agent |

**`VRAMLifecycleManager(contextmanager)`:**

```python
__enter__:
  1. Acquire class-level RLock (120s timeout)
  2. Check free VRAM ≥ required_vram_gb (CPU fallback if insufficient)
  3. Call loader_fn() → model
  4. model.to(device).eval()
  5. torch.no_grad() context

__exit__ / _safe_cleanup:
  1. model.to("cpu")           # Move tensors off GPU
  2. torch.cuda.synchronize()  # Wait for CUDA kernels to finish
  3. del self.model            # Drop Python reference
  4. gc.collect()              # Python GC
  5. torch.cuda.empty_cache()  # Return blocks to CUDA allocator
  6. Release RLock
```

For TPU: uses `xm.mark_step()` instead of `synchronize()`.

**`run_with_vram_cleanup(loader_fn, inference_fn, model_name, required_vram_gb)`:**
- Wraps `VRAMLifecycleManager` with `torch.no_grad()`
- Calls `inference_fn(model)` inside the context
- Called by `agent.py` Segment C GPU loop inside `ThreadPoolExecutor(timeout=60)`

---

### `utils/preprocessing.py` — MediaPipe + CPU-SORT Tracker

**CPU-SORT classes:** `KalmanBoxTracker`, `iou_batch()`, `compute_iou()` (pure NumPy Kalman filter + Hungarian matching via `scipy.optimize.linear_sum_assignment`)

**`KalmanBoxTracker(bbox, track_id)`:**
- 7-state Kalman filter (x,y,w,h,dx,dy,ds) with 4-dimensional measurement (bbox)
- `convert_bbox_to_z(bbox)` → center-scale representation
- `convert_x_to_bbox(x)` → (x1,y1,x2,y2)

**`TrackedFace` (dataclass):**
```
identity_id             int
landmarks               np.ndarray (478, 2) — MediaPipe absolute px coords
trajectory_bboxes       Dict[int, Tuple[int,int,int,int]] — frame_idx → bbox
face_window             Tuple[int, int]  — (start, end) contiguous tracking run
face_crop_224           np.ndarray (224, 224, 3) — sharpest frame crop (v3.0 typo fixed)
face_crop_380           np.ndarray (380, 380, 3) — for SBI
patch_left_periorbital, patch_right_periorbital
patch_nasolabial_left, patch_nasolabial_right
patch_hairline_band, patch_chin_jaw         — 6 anatomical patches
heuristic_flags         List[str]  — [MOTION_BLUR, OCCLUSION, FACE_TOO_SMALL, LOW_LIGHT]
```
`TrackedFace` supports `face.get(key)` and `face[key]` via `__getitem__`.

**`PreprocessResult` (dataclass):**
```
tracked_faces           List[TrackedFace]
frames_30fps            List[np.ndarray]
has_face                bool
max_confidence          float  ← v3.0: tracking_ratio = len(traj)/total_frames (was 1.0)
max_face_area_ratio     float
frames_with_faces_pct   float
heuristic_flags         List[str]
insufficient_temporal_data bool
original_media_type     str   ("image" | "video")
```

**`Preprocessor.process_media(media_path, config)` — main entry:**
```
Image path:
  1. load_image() → RGB np.ndarray
  2. MediaPipe FaceMesh (static_image_mode=True)
  3. Single TrackedFace (face_window = (0, num_frames-1))
  4. max_confidence = 1.0 (deterministic for static image)

Video path:
  1. extract_frames(fps=30) → List[np.ndarray] RGB
  2. For each frame: MediaPipe FaceMesh → detections
  3. CPU-SORT: iou_batch → linear_sum_assignment → track assignment
  4. Build trajectory_bboxes per track_id
  5. Filter: len(trajectory) ≥ min_track_length
  6. face_window = longest contiguous frame run
  7. max_confidence = len(trajectory) / len(frames)  ← v3.0
  8. _select_sharpest_frame → 224×224 crop (cx1:cx2 fix v3.0)
  9. 380×380 crop, 6 anatomical patches
  10. heuristic_flags from blur/area/brightness analysis
```

**`_select_sharpest_frame(frames, trajectory)`:**
- Samples up to `quality_snipe_samples (5)` candidate frames from trajectory
- Laplacian variance for blur → picks sharpest
- v3.0 fix: crop uses `cx2` (clamped) not `x2` (unclamped)

---

### `utils/video.py` — Video I/O

`extract_frames(path, fps=30)` — tries `torchcodec` first, falls back to OpenCV if unavailable or error. Returns `List[np.ndarray]` in RGB order.

`is_video_file(path)` — extension-based check: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.

---

### `utils/image.py` — Image I/O

`load_image(path)` — `cv2.imread` + `cvtColor(BGR→RGB)`. Returns `np.ndarray` shape `(H, W, 3)` uint8 RGB.

`is_image(path)` — extension-based check: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

---

### `utils/logger.py` — Structured Logging

`setup_logger(name)` — returns named logger with consistent format. Write level controlled by `LOG_LEVEL` env var (default `INFO`).

---

## Part 7 — Edge Case Matrix (v3.0 Verified)

| Scenario | Handled? | How |
|---|---|---|
| No face detected | ✅ | `has_face=False` → no-face pipeline (c2pa + dct only) |
| Fully synthetic (Midjourney/DALL-E/FLUX) | ✅ | UnivFD + FreqNet |
| Face swap (DeepFaceLab/FaceSwap) | ✅ | XceptionNet + SBI |
| Reenactment (Face2Face/NeuralTextures) | ✅ | XceptionNet |
| C2PA-signed authentic image | ✅ | C2PA short-circuit → immediate REAL |
| PNG deepfake (no JPEG) | ⚠️ | DCT blind; UnivFD + FreqNet still active |
| CPU tool timeout (> 30s) | ✅ | `FuturesTimeoutError` → `_make_error_result("Timeout after 30s")` |
| GPU tool timeout (> 60s) | ✅ | `FuturesTimeoutError` → `_make_error_result("Timeout after 60s")` |
| VRAM OOM | ✅ | `run_with_vram_cleanup` retries; registry.execute_tool retries once |
| > 50% tools errored | ✅ | `is_degraded=True` propagated in VERDICT event and return dict |
| face_window = (0,0) in rPPG | ✅ | v3.0: ABSTAIN + continue (was: process all frames as noise) |
| max_confidence < 0.60 | ✅ | v3.0: fail face gate → no bio-signal tools run |
| Lambda closure bug in GPU loop | ✅ | v3.0: `make_loader(t)` factory fixes late-binding |
| SBI + fully-synthetic image | ✅ | SBI skips if `visual_score > 0.70` (UnivFD score) |
| CNNDetect weights missing | ⚠️ | FreqNet falls back to FADHook-only |
| UnivFD probe missing | ⚠️ | Random-init probe (score ~0.5, confidence ~0.05) |
| SBI weights missing | ⚠️ | Random-init head — scores unreliable |
| XceptionNet weights missing | ⚠️ | Falls back to zero-head (score ~0.5) |
| C2PA library not installed | ✅ | Graceful abstention (success=True, confidence=0.0) |

---

## Part 8 — Architecture Changes Summary (v1 → v2 → v3.0)

### v1 → v2

| Change | Detail |
|---|---|
| `siglip_adapter_tool.py` deleted | Replaced by `univfd_tool.py` |
| `univfd_tool.py` added | CLIP-ViT-L/14 + linear probe |
| `xception_tool.py` added | XceptionNet FaceForensics++ |
| `freqnet_tool.py` rewritten | FreqNetDual → CNNDetect + FADHook |
| `vram_manager.py` updated | Added `synchronize()` before del model |
| `registry.py` updated | `run_siglip_adapter` → `run_univfd` + `run_xception` |

### v2 → v3.0 (current)

| Change | File | Detail |
|---|---|---|
| `_make_error_result()` DRY factory | `agent.py` | Centralizes all 4 error ToolResult constructors |
| 30s CPU timeouts | `agent.py` | `ThreadPoolExecutor` in `_safe_execute_tool` |
| 60s GPU timeouts | `agent.py` | `ThreadPoolExecutor` wrapping `run_with_vram_cleanup` |
| Face Gate implemented | `agent.py` | 4-dimension routing decision |
| Directional confidence gate | `agent.py` | `(score-0.5)×2` signed math replaces magnitude |
| `decisive_results` filter | `agent.py` | Excludes `|score-0.5| ≤ 0.15` from gate (code uses 0.15, not 0.05) |
| `DEGRADED` propagation | `agent.py` | Now in VERDICT event + return dict |
| Closure-safe GPU lambdas | `agent.py` | `make_loader(t)` / `make_inference(data)` factories |
| Per-model VRAM dict | `agent.py` | `GPU_VRAM_REQUIREMENTS` replaces hardcoded 0.6 |
| Top-level imports | `agent.py` | `torch`, `vram_manager`, `ThreadPoolExecutor` out of loop |
| `max_confidence` tracking ratio | `preprocessing.py` | `len(trajectory)/len(frames)` (was always 1.0) |
| Sharpness crop typo fix | `preprocessing.py` | `cx1:cx2` not `cx1:x2` |
| `face_window=(0,0)` guard | `rppg_tool.py` | ABSTAIN + continue instead of processing all frames |

---

*Last Updated: April 2026 — v3.0 Dual-Pipeline Final Edition*
*Verified against: agent.py, preprocessing.py, rppg_tool.py, ensemble.py, registry.py, thresholds.py, config.py, sbi_tool.py, freqnet_tool.py, univfd_tool.py, xception_tool.py, c2pa_tool.py, dct_tool.py, vram_manager.py*
