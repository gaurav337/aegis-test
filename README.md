<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-RTX_3050+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Phi3:Mini-FF6F00?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Version-5.0_(Illumination_Shield)-8B5CF6?style=for-the-badge" />
</p>

<h1 align="center">🛡️ Aegis-X</h1>
<h3 align="center">Advanced Deepfake Forensic Detection — Dual-Pipeline with Anomaly Shield</h3>

<p align="center">
  <em>A multi-modal, agentic forensic system that orchestrates 10 specialized detection tools — spanning classical physics, frequency analysis, physiological signals, and deep neural networks — through an intelligent CPU→GPU dual-pipeline to deliver explainable, confidence-weighted verdicts on media authenticity.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dual-pipeline-flow">Pipeline</a> •
  <a href="#-tool-manifest">Tools</a> •
  <a href="#-key-changes-v30">What's New</a> •
  <a href="#-web-interface">Web UI</a> •
  <a href="#%EF%B8%8F-configuration">Config</a> •
  <a href="#-benchmarking">Benchmarks</a>
</p>

---

## ✨ Key Features

- **10-Tool Forensic Arsenal** — 6 CPU-bound classical/physics/signal tools + 4 GPU-accelerated neural networks, each targeting an independent manipulation vector
- **Dual-Pipeline with Face Gate** — Media is intelligently routed: a 4-dimension Face Gate decides whether the full bio-signal CPU path runs before GPU inference. **No-face media** skips bio-signal tools and runs the frequency/spectral GPU pipeline directly
- **CPU→GPU Early Stopping Gate** — After the CPU phase, a directional confidence gate (`HALT` / `MINIMAL_GPU` / `FULL_GPU`) prevents unnecessary GPU execution when the CPU already has consensus
- **Three-Pronged Anomaly Shield (v5.0)** — Prevents clean-signal dilution: Suspicion Overdrive (hard GPU max-pool with conflict guard), Borderline Consensus Detection (corroborated weak signals), and GPU Coverage Degradation (penalty when specialists abstain)
- **C2PA AI-Generation Detection (V2)** — Recursively scans all embedded C2PA manifests for AI-generation indicators (IPTC `digitalSourceType`: `trainedAlgorithmicMedia`, `algorithmicMedia`, `compositeSynthetic`). Cryptographic signature validation depth checking (valid/invalid/untrusted/expired). Full provenance chain extraction (creation tool → final tool). When a valid C2PA signature declares AI generation, score = 1.0 (fake) with 0.95 confidence — the strongest possible evidence of synthetic content. Manipulation severity scoring from action history (c2pa.edited, c2pa.retouched, c2pa.composited). Case-insensitive pattern-based keyword matching for AI tool detection (Flux, Ideogram, Leonardo, Kling, Runway, Pika, Sora). Absence of C2PA correctly abstains (score 0.0, confidence 0.0) — no influence on ensemble.
- **GPU Specialist Hierarchy** — GPU neural networks (UnivFD, Xception, SBI, FreqNet) are the "deciders" with weights 0.10–0.18 (registry.py). CPU tools (DCT, Geometry, Corneal, Illumination) are "supporters" with weights 0.04–0.16 (registry.py) — they inform but never override GPU consensus
- **Abstention Transparency** — Tools with `confidence = 0` display `[ABSTAINED] N/A` in the UI and LLM prompt, preventing the "100% Authentic" hallucination trap
- **30s CPU / 60s GPU Per-Tool Timeouts** — `ThreadPoolExecutor`-enforced timeouts on every tool; hung tools never block the pipeline
- **DEGRADED Mode Propagation** — If >50% of tools error out, the verdict dict and SSE events include `"degraded": true` for consumers
- **MediaPipe + CPU-SORT Tracking** — Bidirectional face tracking with Kalman filters; `max_confidence` derived from tracking coverage ratio (not hardcoded 1.0)
- **Explainable AI Verdicts** — Every verdict is grounded in tool-level evidence summaries, passed to Ollama's Phi-3 Mini for natural-language forensic explanations
- **Real-Time Streaming UI** — Glassmorphic web interface with Server-Sent Events (SSE) for live tool-by-tool progress, persistent media preview in results view

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.10 |
| GPU VRAM | 3 GB | 6 GB+ |
| CUDA | 11.8+ | 12.1+ |
| RAM | 8 GB | 16 GB |
| Ollama | Installed with `phi3:mini` pulled | — |

### Automated Setup

```bash
# Clone the repository
git clone https://github.com/gaurav337/aegis-test.git
cd aegis-test

# Run the one-click installer
# (Creates venvs, installs dependencies, downloads weights from Kaggle)
python setup.py
```

> **Note:** You need a [Kaggle API token](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json` for automatic weight downloads.

### Manual Setup

```bash
# 1. Create virtual environments
python3.10 -m venv .venv_main
python3.10 -m venv .venv_gpu

# 2. Install dependencies
.venv_main/bin/pip install -r requirements-main.txt
.venv_gpu/bin/pip install -r requirements-gpu.txt

# 3. Download model weights from Kaggle
# https://www.kaggle.com/datasets/gauravkumarjangid/aegis-pth
# Place them as:
#   models/univfd/probe.pth
#   models/xception/xception_deepfake.pth
#   models/sbi/efficientnet_b4.pth
#   models/freqnet/cnndetect_resnet50.pth

# 4. Download CLIP backbone (auto-downloads on first run, or manually):
#   models/clip-vit-large-patch14/  (from HuggingFace: openai/clip-vit-large-patch14)

# 5. Ensure Ollama is running with Phi-3
ollama pull phi3:mini
ollama serve
```

### Launch

> **⚠️ Important:** Always use the explicit binary path to avoid environment cross-contamination:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv_main/bin/python run_web.py
# Open http://localhost:8000
```

---

## 🏗 Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         WEB INTERFACE                           │
│                   (FastAPI + SSE Streaming)                     │
├─────────────────────────────────────────────────────────────────┤
│                       ForensicAgent                             │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │   .venv_main            │  │   .venv_gpu (via proxy)      │  │
│  │   CPU PHASE             │  │   GPU PHASE                  │  │
│  │                         │  │                              │  │
│  │  [C2PA] → [DCT]        │  │  [FreqNet] → [UnivFD]       │  │
│  │  [rPPG] → [Geometry]   │  │  [Xception] → [SBI*]        │  │
│  │  [Illumin.] → [Corneal]│  │   *face-pipeline only        │  │
│  └──────────┬──────────────┘  └─────────────┬────────────────┘  │
│             │                               │                   │
│          ┌──▼──────────────────────────────▼──┐                │
│          │       CPU→GPU GATE                  │                │
│          │  HALT / MINIMAL_GPU / FULL_GPU       │                │
│          └──────────────────┬──────────────────┘                │
│                             │                                   │
│         ┌───────────────────▼───────────────────┐               │
│         │         EnsembleAggregator             │               │
│         │  (Directional confidence • Suspicion   │               │
│         │   Overdrive • Conflict detection)      │               │
│         └───────────────────┬───────────────────┘               │
│                             │                                   │
│         ┌───────────────────▼───────────────────┐               │
│         │          LLM Synthesis                 │               │
│         │       (Ollama / Phi-3 Mini)            │               │
│         └───────────────────┬───────────────────┘               │
│                             ▼                                   │
│                  { verdict, score, explanation,                 │
│                    degraded } → SSE → Browser                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Dual-Pipeline Flow

### Stage 0 — Preprocessing

```
Media Input (Image / Video)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                  PREPROCESSOR                        │
│  (utils/preprocessing.py)                           │
│                                                     │
│  Video path:                                        │
│    extract_frames() → CPU-SORT Kalman tracking      │
│    Build trajectory_bboxes per tracked face         │
│    face_window = best contiguous run of frames      │
│    max_confidence = tracked_frames / total_frames   │ ← v3.0: was always 1.0
│    224×224 crop + 380×380 crop per identity         │
│    heuristic_flags: MOTION_BLUR, OCCLUSION, etc.   │
│                                                     │
│  Image path:                                        │
│    MediaPipe FaceMesh (static mode)                 │
│    Single face → TrackedFace dataclass              │
│    max_confidence = 1.0 (deterministic)             │
└──────────────────────┬──────────────────────────────┘
                       │ PreprocessResult
                       ▼
```

### Stage 1 — Face Gate (4 Dimensions)

```
                    PreprocessResult
                          │
        ┌─────────────────▼────────────────────┐
        │           FACE GATE                   │
        │  core/agent.py :: analyze()           │
        │                                       │
        │  ① has_face = True?                  │
        │  ② max_confidence ≥ 0.60?            │ ← tracking coverage ratio
        │  ③ max_face_area_ratio ≥ 0.01?       │
        │  ④ frames_with_faces_pct ≥ 0.30?     │
        │                                       │
        │  ALL 4 ✓ → pass_face_gate = True     │
        │  Any ✗  → pass_face_gate = False     │
        └───────┬───────────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  Face Pipeline      No-Face Pipeline
  (bio-signal tools  (frequency/spectral
   + SBI added)       tools only)
```

### Stage 2 — CPU Phase (Segment A)

```
pass_face_gate?
       │
  YES──┤  cpu_tools = [check_c2pa, run_dct, run_rppg,
       │               run_geometry, run_illumination,
       │               run_corneal]
       │
  NO───┤  cpu_tools = [check_c2pa, run_dct]
       │
       │  heuristic_flags may disable some tools:
       │    MOTION_BLUR/OCCLUSION → skip geo/illum/corneal
       │    FACE_TOO_SMALL        → skip corneal
       │    LOW_LIGHT             → skip illum/corneal
       │
       ▼
  For each tool:
    result = _safe_execute_tool(tool_name, input_data, timeout=30)
    ────────────────────────────────────────────────────
    ThreadPoolExecutor → future.result(timeout=30)
    On TimeoutError → _make_error_result("Timeout after 30s")
    On Exception    → _make_error_result(str(e))
    ────────────────────────────────────────────────────
    ensemble.add_result(result)
    yield AgentEvent("TOOL_COMPLETED")

  Special case: C2PA verified?
    → If AI-generated: yield AgentEvent("EARLY_STOP")
      return {"verdict": "FAKE", "score": 0.0, "explanation": ...}
    → If authentic: yield AgentEvent("EARLY_STOP")
      return {"verdict": "REAL", "score": 1.0, "explanation": ...}
```

### Stage 3 — CPU→GPU Gate (Segment B)

```
After CPU phase completes...

    cpu_results = [successful, non-error, non-ABSTAIN results]
    decisive_results = [r for r if |r.score - 0.5| > 0.15]   ← code uses 0.15 (README previously said 0.05)
                                   ↑
                         Filters out neutral-score (0.5)
                         tools that don't know direction

    if len(decisive_results) < 3:          ← code uses 3 (README previously said 2)
        gate_decision = "FULL_GPU"   ← not enough signal
    else:
        Directional Confidence Math:
        ──────────────────────────────
        baseline_weights = {
            run_rppg: 0.35, run_geometry: 0.25,
            run_dct: 0.15, run_illumination: 0.10,
            run_corneal: 0.10, check_c2pa: 0.05
        }
        direction_i   = (score_i - 0.5) × 2  ∈ [-1.0, +1.0]
        agg_direction = Σ (direction_i × confidence_i × normalized_weight_i)
        agg_conf      = |agg_direction|

        Unison = all decisive_results agree on direction (REAL or FAKE)
        Domains = {bio, phys, freq, auth} from contributing tools

        if agg_conf > 0.93 AND unison AND |domains| ≥ 2:
            gate_decision = "HALT"         ← skip GPU entirely
            (Note: domain check is incorporated into unison_agreement flag in code)
        elif agg_conf ≥ 0.80:
            gate_decision = "MINIMAL_GPU"  ← one GPU tool only (UnivFD)
        else:
            gate_decision = "FULL_GPU"     ← full GPU sequence

    yield AgentEvent("GATE_DECISION", {decision, confidence, unison})
```

### Stage 4 — GPU Phase (Segment C)

```
gate_decision ≠ HALT?
       │
       ▼
  Build gpu_sequence:
    pass_face_gate:  [freqnet, univfd, xception, sbi]
    no face:         [freqnet, univfd, xception]
    MINIMAL_GPU:     [univfd]           ← override

  For each tool_name in gpu_sequence:
    tool = registry.get_tool(tool_name)
    req_vram = GPU_VRAM_REQUIREMENTS[tool_name]
    ──────────────────────────────────────────────────
    GPU_VRAM_REQUIREMENTS = {
        "run_freqnet": 0.4 GB
        "run_univfd":  0.6 GB
        "run_xception":0.5 GB
        "run_sbi":     0.8 GB
    }
    ──────────────────────────────────────────────────

    RAW IMAGE FALLBACK (v4.0):               ← no-face pipeline
      Each GPU tool checks tracked_faces first.
      If empty → loads raw image from media_path:
        FreqNet:  resize 224×224, analyze frequency artifacts
        UnivFD:   resize 224×224, CLIP+probe AI detection
        Xception: original res → internal 299×299 resize
        SBI:      NOT RUN (requires face crops by design)

    ThreadPoolExecutor → run_with_vram_cleanup(
        make_loader(tool),
        make_inference(input_data),
        model_name=tool_name,
        required_vram_gb=req_vram,
        timeout=60                ← GPU timeout
    )

    On FuturesTimeoutError → _make_error_result("Timeout after 60s")
    On Exception           → _make_error_result(str(e))
    torch.cuda.empty_cache() in both error paths
```

### Stage 5 — DEGRADED Check + Ensemble + LLM

```
    is_degraded = total_errors / total_results > 0.50
    if is_degraded:
        logger.warning("DEGRADED: >50% tool failures")

    final_score  = ensemble.get_final_score()      ← 1.0 = REAL
    verdict_str  = ensemble.get_verdict()           ← "REAL" / "FAKE"
    explanation  = generate_verdict(...)            ← Ollama LLM

    yield AgentEvent("VERDICT", {
        verdict, score, explanation, degraded       ← v3.0: degraded propagated
    })

    return {
        "verdict": verdict_str,
        "score": final_score,
        "explanation": explanation,
        "degraded": is_degraded                     ← v3.0: consumers can detect
    }
```

---

## 🔬 Tool Manifest

### CPU Tools (`.venv_main` — Zero VRAM)

| Tool | File | Weight | Role | Target Threat | Method |
|---|---|---|---|---|---|
| `check_c2pa` | `c2pa_tool.py` | 0.05 | Gate | Provenance forgery / AI generation | V2: IPTC digitalSourceType + action history severity + cryptographic signature validation + full provenance chain + regex pattern AI keywords |
| `run_dct` | `dct_tool.py` | 0.04 | Supporter | JPEG re-encoding | Double-quantization frequency peaks |
| `run_geometry` | `geometry_tool.py` | 0.08 | Supporter | Anthropometric distortion | IPD, philtrum, vertical thirds (MediaPipe 468-pt) |
| `run_illumination` | `illumination_tool.py` | 0.04 | Supporter | Lighting inconsistency | 2D gradient angular comparison (Sobel), bilateral-filtered skin-masked face luma, per-strip context gradient averaging, nose shadow validation, BGR/RGB auto-detection |
| `run_corneal` | `corneal_tool.py` | 0.04 | Supporter | Missing/mismatched catchlights | Bilateral reflection detection + divergence score |
| `run_rppg` | `rppg_tool.py` | 0.06 | Supporter | Absent biological signal | POS rPPG + FFT (0.7–2.5 Hz cardiac band) ⚠️ |

> **Supporters** inform the weighted average but cannot unilaterally override GPU specialist verdicts. rPPG is video-only.

### GPU Tools (`.venv_gpu` — Sequential VRAM Loading)

| Tool | File | VRAM | Weight | Role | Architecture | Checkpoint |
|---|---|---|---|---|---|---|
| `run_freqnet` | `freqnet_tool.py` | 0.4 GB | 0.10 | Decider | CNNDetect ResNet-50 + FADHook DCT | [Wang et al. CVPR 2020](https://github.com/PeterWang4158/CNNDetect) |
| `run_univfd` | `univfd_tool.py` | 0.6 GB | 0.22 | Decider | CLIP-ViT-L/14 + 4KB linear probe | [Ojha et al. CVPR 2023](https://github.com/ojha-group/UnivFD) |
| `run_xception` | `xception_tool.py` | 0.5 GB | 0.15 | Decider | Xception (FaceForensics++) | [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection) |
| `run_sbi` | `sbi_tool.py` | 0.8 GB | 0.25 | Decider | EfficientNet-B4 + GradCAM | [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) |

> **Peak VRAM:** Never additive. Load → Run → `synchronize → del → gc → empty_cache` between each tool. Max single model = 0.8 GB (SBI).
>
> **No-Face Fallback (v4.0):** When no faces are detected, FreqNet/UnivFD/Xception load the raw image from `media_path` instead of erroring. SBI is excluded from the no-face pipeline.

---

## 🆕 Key Changes v5.0

### C2PA Tool V2 — 6 Critical Fixes

| # | Issue | Fix | Impact |
|---|---|---|---|
| 1 | `is_ai_generated` computed but score always 0.0 | Score = 1.0 (fake) when valid sig + AI declared | Firefly-signed AI images no longer reported as authentic |
| 2 | Valid signature ≠ authentic content | Action history parsed for manipulation severity (c2pa.edited, retouched, composited) | Clean camera capture vs. 10-action Photoshop chain scored differently |
| 3 | C2PA absence treated as neutral | Correctly abstains (0.0/0.0) — ensemble weights asymmetrically | Missing C2PA no longer falsely signals authenticity |
| 4 | Fragile keyword matching | IPTC `digitalSourceType` authoritative + case-insensitive regex patterns (Flux, Ideogram, etc.) | New AI tools detected; "generated thumbnail" false positives eliminated |
| 5 | No signature validation depth | `validation_status` checked for invalid/untrusted/expired | Tampered files with broken signatures flagged as suspicious |
| 6 | Only active manifest's signer checked | Full provenance chain extracted (creation tool → final tool) | AI-generated image re-saved in Photoshop now correctly flagged |

### Illumination Tool V5 — 5 Quality Fixes

| # | Issue | Fix | Impact |
|---|---|---|---|
| 1 | `_ensure_rgb` heuristic triggered on blue-heavy scenes | Multi-channel variance + mean comparison | Ocean/sky photos no longer misinterpreted as BGR |
| 2 | Context vstack created artificial Sobel edge | Per-strip gradient computation with vector averaging | No fake discontinuity between left/right background strips |
| 3 | Scoring used crude left/right ratio | Angular mismatch severity drives penalty (0°–180° scale) | Near-center lighting no longer causes false mismatches |
| 4 | Context luma not bilateral filtered | Bilateral filter applied to context strips (matching face) | Background texture artifacts no longer corrupt context gradient |
| 5 | Skin mask fallback returned unmasked luma | Returns `None` when <100 skin pixels detected | Glasses/heavy makeup faces abstain instead of producing noisy signal |

### Bug Fixes Applied Since Last Commit (v3.0)

| # | File | Change | Impact |
|---|---|---|---|
| 1 | `core/agent.py` | `_safe_execute_tool` now wraps `tool.execute` in `ThreadPoolExecutor(timeout=30)` | CPU tools can no longer hang the pipeline |
| 2 | `core/agent.py` | GPU tools wrapped in `ThreadPoolExecutor(timeout=60)` with per-model VRAM requirements dict | GPU tools timeout properly; no hardcoded 0.6 GB |
| 3 | `core/agent.py` | Confidence aggregation changed to **directional**: `direction = (score - 0.5) × 2 × confidence × weight` | Gate uses directional confidence, not blind magnitude |
| 4 | `core/agent.py` | Unison check filters `score == 0.5` via `decisive_results` list | Neutral scores no longer miscounted as "REAL" |
| 5 | `utils/preprocessing.py` | `max_confidence` derived from `len(trajectory_bboxes) / len(frames)` | Face gate confidence dimension is now meaningful |
| 6 | `core/tools/rppg_tool.py` | `face_window = (0,0)` returns `ABSTAIN` and `continue` instead of processing all frames | rPPG no longer processes noise as signal |
| 7 | `core/agent.py` | `torch` + `vram_manager` + `ThreadPoolExecutor` all imported at top-level | No import-inside-loop overhead |
| 8 | `core/agent.py` | GPU lambda uses `make_loader(t)` / `make_inference(data)` factories | Closure captures tool by value; loop-refactor safe |
| 9 | `core/agent.py` | `is_degraded` propagated to `VERDICT` event AND return dict | Consumers can detect degraded analysis |
| 10 | `core/agent.py` | `_make_error_result(tool_name, msg, start_time)` centralizes all 4 error `ToolResult` constructors | Single source of truth for error shape |
| 11 | `utils/preprocessing.py` | `face_crop = frame[cy1:cy2, cx1:cx2]` (was `cx1:x2` typo) | Sharpness scoring uses fully clamped coordinates |

---

## 📁 File-by-File Reference

### `core/agent.py` — The Orchestrator

The nerve center. Implements `ForensicAgent` as a generator-based orchestrator:

```
ForensicAgent
├── __init__()              — Loads tool registry, ESC, EnsembleAggregator
├── _make_error_result()    — DRY factory for all ToolResult error states
├── _safe_execute_tool()    — CPU tool runner with 30s timeout via ThreadPoolExecutor
└── analyze()               — Main generator pipeline
    ├── Face Gate           — 4-dimension routing decision
    ├── Segment A: CPU      — Heuristic-gated tool execution + C2PA short-circuit
    ├── Segment B: Gate     — Directional conf + unison + domain check → HALT/MINIMAL/FULL
    ├── Segment C: GPU      — Closure-safe VRAM-managed sequential inference (60s timeout)
    ├── DEGRADED check      — flags >50% error rate
    └── Ensemble + LLM      — Final verdict with degraded flag
```

Key constants defined here:
```python
GPU_VRAM_REQUIREMENTS = {
    "run_freqnet": 0.4,   # GB
    "run_univfd":  0.6,
    "run_xception":0.5,
    "run_sbi":     0.8,
}

FACE_GATE_THRESHOLDS = {
    "min_confidence":         0.60,
    "min_face_area_ratio":    0.01,
    "min_frames_with_faces":  0.30,
}
```

---

### `utils/preprocessing.py` — Face Extraction & Tracking

Two-phase processing for video, one-phase for images:

```
PHASE 1 (video only) — Build Trajectories
  MediaPipe FaceMesh (static_image_mode=True per frame)
  CPU-SORT tracker (Kalman + Hungarian matching)
  → established_tracks: Dict[track_id, TrackedFace]
     └── trajectory_bboxes: Dict[frame_idx, (x1,y1,x2,y2)]

PHASE 2 — Extract Crops & Heuristics
  For each track with len(trajectory) ≥ min_track_length:
    face_window = longest contiguous run of frames
    max_confidence = len(trajectory) / total_frames   ← v3.0 fix
    Select sharpest frame (Laplacian variance)
    Generate face_crop_224, face_crop_380
    Extract 6 anatomical patches (periorbital, nasolabial, hairline, jaw)
    Set heuristic_flags: MOTION_BLUR, OCCLUSION, FACE_TOO_SMALL, LOW_LIGHT

PreprocessResult fields (relevant to gate):
  has_face                bool
  max_confidence          float  ← tracking coverage ratio [0–1]
  max_face_area_ratio     float  ← peak face/frame area ratio
  frames_with_faces_pct   float  ← % frames containing any face
  heuristic_flags         List[str]
  insufficient_temporal_data bool
```

**TrackedFace dataclass** (dict-compatible via `get()` and `__getitem__`):
```python
TrackedFace:
  identity_id       int
  landmarks         np.ndarray  (478, 2)
  trajectory_bboxes Dict[int, Tuple[int,int,int,int]]
  face_window       Tuple[int, int]   ← (start_frame, end_frame)
  face_crop_224     np.ndarray
  face_crop_380     np.ndarray
  patch_left_periorbital, patch_right_periorbital
  patch_nasolabial_left, patch_nasolabial_right
  patch_hairline_band, patch_chin_jaw
  heuristic_flags   List[str]
```

---

### `core/tools/rppg_tool.py` — Remote Photoplethysmography

Video-only physiological liveness detector:

```
_run_inference()
  ├── Skip if image (returns SKIPPED)
  ├── Skip if frames < RPPG_MIN_FRAMES (returns ABSTAIN)
  ├── Backup face check if tracked_faces empty (_lightweight_face_check)
  │
  └── For each tracked face:
        face_window check                      ← v3.0 fix
          (0,0) → append ABSTAIN, continue     ← was: process all frames as noise
        Slice frames[start:end]
        Remap trajectory keys to relative indices

        Extract POS signals for 3 ROIs:
          forehead, left_cheek, right_cheek
           Hair occlusion guard (Laplacian variance > RPPG_HAIR_OCCLUSION_VARIANCE → ABSTAIN) ⚠️ threshold=0.25 (see Known Issues)
          Darkness guard (mean < 50 → return None)

        _evaluate_liveness():
          AMBIGUOUS     (hair occlusion)
          NO_PULSE      (flat signal across all ROIs)
          SYNTHETIC_FLATLINE (< 2 ROIs with variance)
          WEAK_PULSE_FAILED  (only 1 ROI passes)
          INCOHERENT    (peaks don't synchronize)
          PULSE_PRESENT (≥2 ROIs coherent within 0.5 Hz — RPPG_COHERENCE_THRESHOLD_HZ)

Best face by confidence wins. Returns ToolResult with liveness_label.
```

---

### `utils/ensemble.py` — Score Aggregation (v4.0 — Three-Pronged Anomaly Shield)

```
calculate_ensemble_score()
  Step 1: Extract context (DCT peak ratio, compression flag)
  Step 2: C2PA override check (with visual corroboration guard)
  Step 3: Route each tool via _route()
    each tool returns (contribution, effective_weight)
    Routing rules:
      rPPG    → threshold gated (uses implied prob only at extremes)
      SBI     → blind spot below 0.50; mid-band uses UnivFD context
      FreqNet → blind spot below 0.45; compression discount
      UnivFD/Xception → direct score × weight
      CPU tools → low weight (0.04–0.16 registry.py), supporters only
  Step 4: Three-Pronged Anomaly Scoring
    PRONG 1 — Suspicion Overdrive:
      max_gpu_prob = max(GPU specialist implied probs)
      if max_gpu_prob > 0.70:
        GPU spread check (max - min > 0.30 → conflict → use base avg)
        else: fake_score = max_gpu_prob (hard max-pooling)
    PRONG 2 — Borderline Consensus:
      if ≥2 GPU specialists in [0.35, 0.55] zone:
        consensus = mean × 1.25 boost
    PRONG 3 — GPU Coverage Degradation:
      per-abstained GPU specialist: +10% multiplicative boost
      disabled when GPU conflict detected (prevents double-penalty)
    Final: fake_score = max(base_avg, anomaly_anchor, consensus)
           × gpu_degradation_boost
    GPU conflict guard: if spread > 0.30, anchors disabled → pure avg

ensemble_score = 1.0 - fake_score   (REAL probability)
verdict = "FAKE" if ensemble_score ≤ 0.50 else "REAL"
```

---

### `utils/vram_manager.py` — GPU Memory Lifecycle

```
VRAMLifecycleManager (context manager)
  __enter__:
    Acquire global RLock (120s timeout)
    Check VRAM availability (required_gb threshold)
    CPU fallback if insufficient
    Load model, move to device, set .eval()
  __exit__ / _safe_cleanup:
    model.to("cpu")
    torch.cuda.synchronize()
    del self.model
    _cleanup_device_memory() → empty_cache / mark_step (TPU)
    Release RLock

Hardware priority: TPU → CUDA → MPS → CPU

run_with_vram_cleanup(model_loader, inference_fn, model_name, required_vram_gb)
  → Wraps VRAMLifecycleManager + torch.no_grad()
  → Called from GPU phase in agent.py via ThreadPoolExecutor(timeout=60)
```

---

### `core/early_stopping.py` — Evidential Subjective Logic

The ESL controller is still instantiated but the v3.0 architecture primarily uses the **inline CPU→GPU gate** in `agent.py` for routing decisions. The ESC is available for future re-integration:

```
EarlyStoppingController.evaluate():
  1. C2PA hardware lock → immediate HALT
  2. Validate tool names against registry
  3. Compute weighted_sum, weights_run, weights_pending
  4. Evidential Subjective Logic:
     e_fake = Σ(weight_i × max(0, score_i - 0.5) × 2)
     e_real = Σ(weight_i × max(0, 0.5 - score_i) × 2)
     conflict_ratio = min(e_fake, e_real) / max(e_fake, e_real)
     if conflict_ratio > 0.35 → CONTINUE (adversarial conflict)
  5. Mathematical bounds:
     max_possible = (sum + 1.0 × pending_weight) / total_viable
     min_possible = (sum + 0.0 × pending_weight) / total_viable
  6. Locked FAKE check → HALT if min_possible > real_threshold
  7. Default → CONTINUE

Note: HALT_LOCKED_REAL is intentionally disabled — system always runs
GPU tools before declaring REAL to prevent sophisticated fake bypass.
```

---

## 🌐 Web Interface

The web UI is a premium glassmorphic dark-mode interface built with vanilla HTML/CSS/JS and served by FastAPI.

### Features

- **Drag & Drop Upload** — Accepts JPEG, PNG, WebP, and video files
- **Real-Time SSE Streaming** — Watch each tool execute live with progress indicators
- **GATE_DECISION Event** — UI shows whether GPU was triggered, minimized, or halted
- **DEGRADED Banner** — Displayed when `degraded: true` in VERDICT event
- **Verdict Banner** — Color-coded AUTHENTIC (green) / TAMPERED (red) with composite confidence score
- **Tool Cards** — Score, confidence, and evidence summary per forensic tool
- **Agent Synthesis Panel** — LLM-generated natural language forensic explanation

### SSE Event Stream

| Event | When | Payload |
|---|---|---|
| `PIPELINE_SELECTED` | After face gate | `{face_pipeline: bool}` |
| `TOOL_STARTED` | Before each tool | `{tool_name}` |
| `TOOL_COMPLETED` | After each tool | `{success, confidence}` |
| `EARLY_STOP` | C2PA verified | `{reason: "C2PA_VERIFIED"}` |
| `GATE_DECISION` | After CPU phase | `{decision, confidence, unison}` |
| `llm_start` | Before LLM | — |
| `VERDICT` | Final | `{verdict, score, explanation, degraded}` |

### Scoring Convention

| Score | Convention | Meaning |
|---|---|---|
| `0.0` | AUTHENTIC | Tool found no manipulations |
| `0.5` | NEUTRAL / ERROR | Tool abstained or errored |
| `1.0` | TAMPERED | Tool detected manipulation artifacts |

> **Critical:** `ensemble_score` is the **REAL probability** (1.0 = authentic, 0.0 = fake). The individual tool `score` field is the **FAKE probability**.

---

## ⚙️ Configuration

### Environment Variables

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_MODEL_DIR` | `models/` | Root directory for all model weight files |
| `AEGIS_DEVICE` | `auto` | Force `cuda`, `cpu`, or `auto` detect |
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama LLM server URL |
| `OLLAMA_MODEL` | `phi3:mini` | LLM model for verdict synthesis |
| `AEGIS_VRAM_THRESHOLD` | `3.5` | Minimum free VRAM (GB) to attempt GPU tools |
| `LLM_TEMPERATURE` | `0.1` | Low temperature for deterministic forensic output |
| `LLM_MAX_TOKENS` | `1024` | Maximum tokens for LLM response |

### Thresholds (`utils/thresholds.py` — Single Source of Truth)

| Threshold | Value | Purpose |
|---|---|---|
| `ENSEMBLE_REAL_THRESHOLD` | 0.50 | `ensemble_score` ≤ this → FAKE verdict |
| `ENSEMBLE_FAKE_THRESHOLD` | 0.60 | Score ≥ this → confident FAKE |
| `SUSPICION_OVERRIDE_THRESHOLD` | 0.70 | Hard max-pooling trigger for GPU specialists |
| `BORDERLINE_CONSENSUS_LOW` | 0.35 | Lower bound of borderline detection zone |
| `BORDERLINE_CONSENSUS_HIGH` | 0.55 | Upper bound of borderline detection zone |
| `BORDERLINE_CONSENSUS_BOOST` | 1.25 | Corroboration multiplier for consensus |
| `GPU_COVERAGE_DEGRADATION_FACTOR` | 0.10 | Per-abstained GPU specialist penalty |
| `CONFLICT_STD_THRESHOLD` | 0.20 | Tool disagreement level flagged as conflict |
| `SBI_BLIND_SPOT_THRESHOLD` | 0.50 | SBI score below this → abstain |
| `FREQNET_BLIND_SPOT_THRESHOLD` | 0.45 | FreqNet score below this → abstain |
| `RPPG_CARDIAC_BAND_MIN_HZ` | 0.7 | Lower bound of cardiac frequency (42 BPM) |
| `RPPG_CARDIAC_BAND_MAX_HZ` | 2.5 | Upper bound of cardiac frequency (150 BPM) ⚠️ README previously stated 4.0 Hz |

---

## 📁 Project Structure

```
aegis-x/
├── run_web.py                  # Entry point — FastAPI + SSE streaming server
├── setup.py                    # One-click installer (venvs + Kaggle weights)
├── verify_tools.py             # Diagnostic: test all tools individually
│
├── core/
│   ├── agent.py                # ★ ForensicAgent — Full dual-pipeline orchestration
│   │                           #   Face Gate → CPU → CPU/GPU Gate → GPU → Ensemble → LLM
│   ├── early_stopping.py       # Evidential Subjective Logic gating (ESC)
│   ├── forensic_summary.py     # LLM prompt builder with grounded evidence
│   ├── llm.py                  # Ollama HTTP client bridge
│   ├── memory.py               # SQLite-backed case memory system
│   ├── config.py               # Typed dataclass configuration hierarchy
│   ├── data_types.py           # ToolResult contract (score/fake_score alias)
│   ├── base_tool.py            # Abstract base class for all forensic tools
│   ├── subprocess_proxy.py     # Bridge: .venv_main ↔ .venv_gpu
│   ├── subprocess_worker.py    # GPU worker process (runs in .venv_gpu)
│   ├── exceptions.py           # Custom exception hierarchy
│   └── tools/
│       ├── registry.py         # Tool manifest, weights, trust tiers, dispatch
│       ├── c2pa_tool.py        # [CPU] C2PA V2: AI provenance + signature validation + chain
│       ├── dct_tool.py         # [CPU] JPEG double-quantization detection
│       ├── geometry_tool.py    # [CPU] 468-landmark anthropometric ratio analysis
│       ├── illumination_tool.py# [CPU] 2D angular gradient lighting analysis (V5)
│       ├── corneal_tool.py     # [CPU] Catchlight reflection divergence scoring
│       ├── rppg_tool.py        # [CPU] POS rPPG + FFT cardiac liveness (video)
│       ├── univfd_tool.py      # [GPU] CLIP-ViT-L/14 + linear probe (CVPR 2023)
│       ├── xception_tool.py    # [GPU] XceptionNet (FaceForensics++)
│       ├── sbi_tool.py         # [GPU] Self-Blended Images + GradCAM
│       ├── freqnet_tool.py     # [GPU] CNNDetect + FADHook frequency fusion
│       └── freqnet/
│           ├── preprocessor.py # DCT + spatial preprocessing
│           ├── fad_hook.py     # Frequency Artifact Detection hooks
│           └── calibration.py  # Z-score baseline calibration
│
├── utils/
│   ├── vram_manager.py         # GPU lifecycle (sync→del→gc→empty_cache) + RLock
│   ├── preprocessing.py        # ★ MediaPipe + CPU-SORT tracking + face crops
│   ├── ensemble.py             # ★ Three-Pronged Anomaly Shield + Suspicion Overdrive + C2PA guard
│   ├── thresholds.py           # ★ Central numeric constants (single source of truth)
│   ├── ollama_client.py        # HTTP client for local Ollama server
│   ├── video.py                # Video I/O (torchcodec → OpenCV fallback)
│   ├── image.py                # Image I/O helpers
│   └── logger.py               # Structured logging setup
│
├── web/
│   ├── index.html              # Glassmorphic dark-mode UI
│   ├── style.css               # Full design system (Outfit font, animations)
│   └── script.js               # SSE client, tool card rendering, verdict display
│
├── models/                     # .gitignored — downloaded via setup.py
│   ├── clip-vit-large-patch14/ # CLIP ViT-L/14 backbone (~890 MB)
│   ├── univfd/probe.pth        # Linear probe (4 KB)
│   ├── xception/xception_deepfake.pth  # XceptionNet (80 MB)
│   ├── sbi/efficientnet_b4.pth         # EfficientNet-B4 (135 MB)
│   └── freqnet/cnndetect_resnet50.pth  # CNNDetect ResNet-50 (270 MB)
│
├── requirements-main.txt       # Main env: FastAPI, MediaPipe, OpenCV, scipy
├── requirements-gpu.txt        # GPU env: PyTorch, Transformers, timm
├── .env.example                # Environment variable template
└── .gitignore
```

> ★ = Most heavily modified files in v3.0

---

## 🧪 Diagnostics

```bash
# Test with a specific image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py --image path/to/face.jpg

# Test with default image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py
```

---

## 📊 Benchmarking Datasets

| Dataset | Year | Threat Class | Primary Tools |
|---|---|---|---|
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | 2023 | Text-to-image AI (Midjourney, SD, DALL-E) | `run_univfd` |
| [ArtiFact](https://github.com/awsaf49/artifact) | 2023 | Multi-generator + real-world compression | `run_univfd`, `run_freqnet` |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | 2019 | Face-swap and reenactment | `run_xception`, `run_sbi` |
| [ForgeryNet](https://github.com/yinanhe/forgerynet) | 2021 | Surgical face edits, morphs | `run_sbi` |
| [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild) | 2020 | Internet-scraped deepfakes | Full pipeline |
| [DiffusionForensics](https://github.com/ZhendongWang6/DIRE) | 2023 | Diffusion outputs with heavy compression | `run_freqnet` |

---

## 🔑 Model Weights

**Dataset:** [kaggle.com/datasets/gauravkumarjangid/aegis-pth](https://www.kaggle.com/datasets/gauravkumarjangid/aegis-pth)

| Weight File | Size | Model | Source Paper |
|---|---|---|---|
| `probe.pth` | 4 KB | UnivFD linear probe | Ojha et al., *Towards Universal Fake Image Detectors*, CVPR 2023 |
| `xception_deepfake.pth` | 80 MB | Xception (FaceForensics++) | Rössler et al., *FaceForensics++*, ICCV 2019 |
| `efficientnet_b4.pth` | 135 MB | SBI EfficientNet-B4 | Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images*, CVPR 2022 |
| `cnndetect_resnet50.pth` | 270 MB | CNNDetect ResNet-50 | Wang et al., *CNN-generated images are surprisingly easy to spot*, CVPR 2020 |

CLIP-ViT-L/14 (~890 MB) auto-downloads from HuggingFace on first run.

---

## 📚 References

```bibtex
@inproceedings{ojha2023universal,
  title={Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle={CVPR}, year={2023}
}

@inproceedings{shiohara2022sbi,
  title={Detecting Deepfakes with Self-Blended Images},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={CVPR}, year={2022}
}

@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={ICCV}, year={2019}
}

@inproceedings{wang2020cnndetect,
  title={CNN-generated images are surprisingly easy to spot... for now},
  author={Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A},
  booktitle={CVPR}, year={2020}
}
```

---

## ⚠️ Known Issues & Discrepancies (Verified April 2026)

> These are documented, verified discrepancies between code, configuration, and documentation. They are tracked here for transparency.

### Critical Bugs

| # | File | Issue | Impact | Status |
|---|---|---|---|---|
| 1 | `core/llm.py:100` | `logger` is never imported — `NameError` on LLM timeout (queue.Empty after 300s) | Pipeline crashes if LLM times out | **Needs fix** |
| 2 | `utils/thresholds.py:121` | `RPPG_HAIR_OCCLUSION_VARIANCE = 0.25` but README/code comments say 35.0 — Laplacian variance is typically 30-200+ | Nearly every frame triggers false hair occlusion → rPPG abstains unnecessarily | **Needs fix** |
| 3 | `utils/thresholds.py:120` | `RPPG_CARDIAC_BAND_MAX_HZ = 2.5` (150 BPM) but README documents 4.0 (240 BPM) | Cardiac peaks above 150 BPM are missed | **Needs fix** |
| 4 | `run_web.py:30` | `file.filename` used directly without sanitization — path traversal possible | Security vulnerability: `../../etc/passwd` could write outside upload dir | **Needs fix** |

### Configuration Inconsistencies

| Setting | `config.py` default | `thresholds.py` value | Notes |
|---|---|---|---|
| `AGENT_MAX_RETRIES` | 2 | 3 | Two separate default sources |
| `LLM_MAX_TOKENS` | 1024 | 512 | Two separate default sources |

### Documentation vs. Code Discrepancies

| Item | README says | Code actually uses | File:Line |
|---|---|---|---|
| Decisive threshold | `\|score - 0.5\| > 0.05` | `\|score - 0.5\| > 0.15` | `agent.py:168` |
| Gate decisive count | `< 2` → FULL_GPU | `< 3` → FULL_GPU | `agent.py:174` |
| rPPG cardiac band max | 4.0 Hz (240 BPM) | 2.5 Hz (150 BPM) | `thresholds.py:120` |
| rPPG hair occlusion | 35.0 | 0.25 | `thresholds.py:121` |

### Weight Mismatches (registry.py vs. thresholds.py)

**registry.py is the runtime source of truth** — these are the actual weights used by the tool registry and EarlyStoppingController. The `thresholds.py` weights are used by the ensemble scorer. These differ:

| Tool | `registry.py` (runtime) | `thresholds.py` (ensemble) |
|---|---|---|
| `check_c2pa` | 0.04 | 0.05 |
| `run_dct` | 0.06 | 0.04 |
| `run_rppg` | 0.05 | 0.06 |
| `run_geometry` | 0.16 | 0.08 |
| `run_illumination` | 0.04 | 0.04 |
| `run_corneal` | 0.06 | 0.04 |
| `run_univfd` | 0.18 | 0.22 |
| `run_xception` | 0.13 | 0.15 |
| `run_sbi` | 0.18 | 0.25 |
| `run_freqnet` | 0.10 | 0.10 |
| **Total** | **1.00** | **1.03** |

### Security Notes

- No authentication or rate limiting on `/api/analyze`
- No filename sanitization (path traversal risk)
- No file cleanup after analysis (disk exhaustion risk)
- No upload collision handling (same filename overwrites)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🔬 forensic rigor and ⚡ engineering precision.<br>
  <em>v5.0 — C2PA V2 (AI provenance + signature validation) · Illumination V5 (angular gradient + skin mask + bilateral filter) · Anomaly Shield v5.0</em>
</p>
