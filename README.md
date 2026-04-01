<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-RTX_3050+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Phi3:Mini-FF6F00?style=for-the-badge&logo=meta&logoColor=white" />
</p>

<h1 align="center">🛡️ Aegis-X</h1>
<h3 align="center">Advanced Deepfake Forensic Detection Pipeline</h3>

<p align="center">
  <em>A multi-modal, agentic forensic system that orchestrates 8 specialized detection tools — spanning classical physics, frequency analysis, and deep neural networks — to deliver explainable, confidence-weighted verdicts on media authenticity.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tool-manifest">Tools</a> •
  <a href="#-pipeline-flow">Pipeline</a> •
  <a href="#-web-interface">Web UI</a> •
  <a href="#%EF%B8%8F-configuration">Config</a> •
  <a href="#-benchmarking">Benchmarks</a>
</p>

---

## ✨ Key Features

- **10-Tool Forensic Arsenal** — 4 CPU-bound classical/physics tools + 4 GPU-accelerated neural networks, each targeting a distinct manipulation vector
- **Agentic Orchestration** — A `ForensicAgent` dynamically sequences tools, applies early stopping via Evidential Subjective Logic, and synthesizes results through a local LLM
- **VRAM-Safe on 4GB GPUs** — Sequential GPU model loading with full lifecycle management (`synchronize → del → gc → empty_cache`). Peak VRAM = single largest model (~1.8 GB), never additive
- **Explainable AI Verdicts** — Every verdict is grounded in tool-level evidence summaries, passed to Ollama's Phi-3 Mini for natural-language forensic explanations
- **Real-Time Streaming UI** — Glassmorphic web interface with Server-Sent Events (SSE) for live tool-by-tool progress updates
- **Dual Virtual Environment** — Lightweight main env (FastAPI + MediaPipe) isolated from heavy GPU env (PyTorch + Transformers), bridged via subprocess proxy
- **One-Click Setup** — Automated `setup.py` creates both environments, downloads model weights from Kaggle, and places them in the correct directories

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

```bash
# Start the web server
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_main/bin/python run_web.py

# Open in browser
# http://localhost:8000
```

---

## 🏗 Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                     WEB INTERFACE                       │
│              (FastAPI + SSE Streaming)                   │
├─────────────────────────────────────────────────────────┤
│                   ForensicAgent                         │
│         (Orchestration + Early Stopping)                │
├────────────────────┬────────────────────────────────────┤
│   CPU PHASE        │         GPU PHASE                  │
│   (.venv_main)     │         (.venv_gpu via proxy)      │
│                    │                                    │
│  ┌─────────────┐   │   ┌──────────┐  ┌──────────┐      │
│  │  Geometry   │   │   │  UnivFD  │  │ Xception │      │
│  │  Illumin.   │   │   │  (CLIP)  │  │ (FF++)   │      │
│  │  Corneal    │   │   ├──────────┤  ├──────────┤      │
│  │  rPPG       │   │   │   SBI    │  │ FreqNet  │      │
│  └─────────────┘   │   │(Eff-B4)  │  │(ResNet50)│      │
│                    │   └──────────┘  └──────────┘      │
├────────────────────┴────────────────────────────────────┤
│              EnsembleAggregator                         │
│     (Weighted scoring + Conflict detection)             │
├─────────────────────────────────────────────────────────┤
│              LLM Synthesis (Ollama / Phi-3)             │
│        (Natural language forensic explanation)          │
└─────────────────────────────────────────────────────────┘
```

### Dual-Environment Design

Aegis-X uses a **subprocess proxy architecture** to isolate the lightweight web server from heavy PyTorch/CUDA dependencies:

| Environment | Purpose | Key Dependencies |
|---|---|---|
| `.venv_main` | Web server, preprocessing, CPU tools, orchestration | FastAPI, MediaPipe, OpenCV, scipy |
| `.venv_gpu` | GPU model inference (invoked via subprocess) | PyTorch, Transformers, timm |

The `SubprocessProxy` in `core/subprocess_proxy.py` bridges both environments — the main process sends serialized input to a GPU worker process, which loads models, runs inference, and returns `ToolResult` objects.

---

## 🔬 Tool Manifest

### CPU Tools (Zero VRAM)

| Tool | Weight | Trust | Target Threat | Method |
|---|---|---|---|---|
| `run_geometry` | 0.18 | Tier 3 | Anthropometric distortions | MediaPipe landmark ratios (IPD, philtrum, eye symmetry, nose width, mouth width, vertical thirds) |
| `run_corneal` | 0.07 | Tier 2 | Missing/inconsistent catchlights | Corneal reflection detection + bilateral divergence scoring |
| `run_rppg` | 0.06 | Tier 2 | Absent biological signal | Remote photoplethysmography — FFT cardiac band extraction (video only) |
| `run_illumination` | 0.05 | Tier 1 | Lighting inconsistency | Gradient-based directional light estimation across face regions |

### GPU Tools (Sequential Loading)

| Tool | Weight | Trust | Target Threat | Architecture | Checkpoint |
|---|---|---|---|---|---|
| `run_univfd` | 0.20 | Tier 3 | Generative AI (FLUX, Midjourney, DALL-E, Stable Diffusion) | CLIP-ViT-L/14 + 4KB linear probe | [Ojha et al. CVPR 2023](https://github.com/ojha-group/UnivFD) |
| `run_sbi` | 0.20 | Tier 3 | Face compositing / blend boundaries | EfficientNet-B4 + GradCAM | [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) |
| `run_xception` | 0.15 | Tier 2 | Face-swap / reenactment (DeepFaceLab, Face2Face) | Xception (FaceForensics++) | [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection) |
| `run_freqnet` | 0.09 | Tier 1 | GAN/diffusion frequency artifacts | CNNDetect ResNet-50 + FADHook DCT energy | [PeterWang4158/CNNDetect](https://github.com/PeterWang4158/CNNDetect) |

> **Ensemble weight sum = 1.00** — Weights are centralized in `utils/thresholds.py` and enforced by the `ToolRegistry`.

---

## 🔄 Pipeline Flow

```
Media Input (JPEG / PNG / WebP)
        │
        ▼
┌──────────────────────────────────────┐
│         PREPROCESSOR                 │
│  MediaPipe face mesh extraction      │
│  224×224 + 380×380 crop generation   │
│  Quality-snipe frame sampling        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         CPU PHASE (Zero VRAM)        │
│                                      │
│  Geometry ──► Illumination           │
│  Corneal  ──► rPPG (video only)      │
│                                      │
│  Each tool returns ToolResult with   │
│  score ∈ [0,1], confidence, and      │
│  evidence_summary string             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│    EARLY STOPPING CONTROLLER         │
│                                      │
│  Evidential Subjective Logic:        │
│  • Conflict detection (min/max       │
│    evidence ratio)                   │
│  • Mathematical bound checking       │
│  • HALT if locked FAKE               │
│  • Always runs GPU for REAL claims   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│     GPU PHASE (Sequential Load)      │
│                                      │
│  UnivFD ──► Xception                 │
│  SBI    ──► FreqNet                  │
│                                      │
│  VRAMLifecycleManager ensures:       │
│  synchronize() → del → gc →          │
│  empty_cache() between each tool     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      ENSEMBLE AGGREGATOR             │
│                                      │
│  • Weighted average with routing     │
│  • Compression discount (DCT→SBI)   │
│  • Blind-spot abstention             │
│  • Conflict std detection            │
│  • C2PA override with visual check   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      LLM SYNTHESIS (Ollama)          │
│                                      │
│  Phi-3 Mini receives:               │
│  • All tool scores + evidence        │
│  • Scoring logic (0=Real, 1=Fake)    │
│  • Instruction to explain in         │
│    plain language                     │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌─────────────┐
        │   VERDICT    │
        │              │
        │  REAL / FAKE │
        │  + score %   │
        │  + explain   │
        └─────────────┘
```

---

## 🌐 Web Interface

The web UI is a premium glassmorphic dark-mode interface built with vanilla HTML/CSS/JS and served by FastAPI.

### Features

- **Drag & Drop Upload** — Accepts JPEG, PNG, and WebP
- **Real-Time SSE Streaming** — Watch each tool execute live with progress indicators
- **Verdict Banner** — Color-coded AUTHENTIC (green) / TAMPERED (red) with composite confidence score
- **Tool Cards** — Individual score, confidence, and evidence summary for each forensic tool
- **Agent Synthesis Panel** — LLM-generated natural language explanation of all findings
- **Responsive Design** — Animated backgrounds, hover effects, smooth transitions

### Scoring Interpretation

| Score Range | UI Status | Meaning |
|---|---|---|
| `0.00 – 0.54` | ✅ CLEAR | Tool found no evidence of manipulation |
| `0.55 – 1.00` | ⚠️ SUSPICIOUS | Tool detected potential manipulation artifacts |
| Error | ❌ ERROR | Tool encountered an exception (VRAM, missing weights, etc.) |

> **Critical:** Scores follow forensic convention — `0.0 = Authentic`, `1.0 = Tampered`. Lower is better.

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

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

### Thresholds

All numeric thresholds are centralized in [`utils/thresholds.py`](utils/thresholds.py) — the single source of truth. Key values:

| Threshold | Value | Purpose |
|---|---|---|
| `ENSEMBLE_REAL_THRESHOLD` | 0.15 | Below this → lean towards REAL |
| `ENSEMBLE_FAKE_THRESHOLD` | 0.85 | Above this → lean towards FAKE |
| `EARLY_STOP_CONFIDENCE` | 0.85 | Confidence required for early halt |
| `SBI_FAKE_THRESHOLD` | 0.60 | SBI score above this = blend detected |
| `UNIVFD_FAKE_THRESHOLD` | 0.60 | UnivFD score above this = generative AI detected |
| `CONFLICT_STD_THRESHOLD` | 0.20 | Tool disagreement level that blocks early stopping |

---

## 📁 Project Structure

```
aegis-test/
├── run_web.py                  # Entry point — FastAPI + SSE streaming server
├── setup.py                    # One-click installer (venvs + Kaggle weights)
├── verify_tools.py             # Diagnostic: test all tools individually
│
├── core/
│   ├── agent.py                # ForensicAgent — CPU→GPU→Ensemble→LLM orchestration
│   ├── early_stopping.py       # Evidential Subjective Logic gating
│   ├── forensic_summary.py     # LLM prompt builder with grounded evidence
│   ├── llm.py                  # Ollama HTTP client bridge
│   ├── memory.py               # SQLite-backed case memory system
│   ├── config.py               # Typed dataclass configuration hierarchy
│   ├── data_types.py           # ToolResult contract + data structures
│   ├── base_tool.py            # Abstract base class for all forensic tools
│   ├── subprocess_proxy.py     # Bridge: .venv_main ↔ .venv_gpu
│   ├── subprocess_worker.py    # GPU worker process (runs in .venv_gpu)
│   ├── exceptions.py           # Custom exception hierarchy
│   └── tools/
│       ├── registry.py         # Tool manifest, weights, trust tiers, dispatch
│       ├── geometry_tool.py    # [CPU] Anthropometric ratio analysis
│       ├── illumination_tool.py# [CPU] Directional lighting consistency
│       ├── corneal_tool.py     # [CPU] Catchlight reflection analysis
│       ├── rppg_tool.py        # [CPU] Remote photoplethysmography
│       ├── c2pa_tool.py        # [CPU] C2PA provenance verification
│       ├── dct_tool.py         # [CPU] JPEG double-quantization detection
│       ├── univfd_tool.py      # [GPU] CLIP-ViT + linear probe (CVPR 2023)
│       ├── xception_tool.py    # [GPU] XceptionNet (FaceForensics++)
│       ├── sbi_tool.py         # [GPU] Self-Blended Images + GradCAM
│       ├── freqnet_tool.py     # [GPU] CNNDetect + FADHook frequency fusion
│       └── freqnet/
│           ├── preprocessor.py # DCT + Spatial preprocessing (pure math)
│           ├── fad_hook.py     # Frequency Artifact Detection hooks
│           └── calibration.py  # Z-score baseline calibration manager
│
├── utils/
│   ├── vram_manager.py         # GPU lifecycle (sync→del→gc→empty_cache)
│   ├── preprocessing.py        # MediaPipe face extraction + frame sampling
│   ├── ensemble.py             # Weighted scoring + conflict detection + EMA
│   ├── thresholds.py           # Central numeric constants (single source of truth)
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
├── requirements-main.txt       # Main env: FastAPI, MediaPipe, OpenCV
├── requirements-gpu.txt        # GPU env: PyTorch, Transformers, timm
├── .env.example                # Environment variable template
└── .gitignore
```

---

## 🧪 Diagnostics

Run the built-in diagnostic tool to verify all CPU and GPU tools are functioning:

```bash
# Test with a specific image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py --image path/to/face.jpg

# Test with default image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py
```

This will sequentially execute every tool and report:
- ✅ Whether weights loaded correctly (not random initialization)
- ✅ Score and confidence values
- ✅ Evidence summary strings
- ✅ Any fatal exceptions

---

## 📊 Benchmarking Datasets

To evaluate the pipeline against the modern deepfake threat landscape:

| Dataset | Year | Threat Class | Primary Tools |
|---|---|---|---|
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | 2023 | Text-to-image AI (Midjourney, SD, DALL-E) | `run_univfd` |
| [ArtiFact](https://github.com/awsaf49/artifact) | 2023 | Multi-generator with real-world compression | `run_univfd`, `run_freqnet` |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | 2019 | Face-swap and reenactment | `run_xception`, `run_sbi` |
| [ForgeryNet](https://github.com/yinanhe/forgerynet) | 2021 | Surgical face edits, morphs | `run_sbi` |
| [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild) | 2020 | Internet-scraped deepfakes | Full pipeline |
| [DiffusionForensics](https://github.com/ZhendongWang6/DIRE) | 2023 | Diffusion outputs with heavy compression | `run_freqnet` |

---

## 🔑 Model Weights

All model weights are hosted on Kaggle and automatically downloaded by `setup.py`:

**Dataset:** [kaggle.com/datasets/gauravkumarjangid/aegis-pth](https://www.kaggle.com/datasets/gauravkumarjangid/aegis-pth)

| Weight File | Size | Model | Source Paper |
|---|---|---|---|
| `probe.pth` | 4 KB | UnivFD linear probe | Ojha et al., *Towards Universal Fake Image Detectors*, CVPR 2023 |
| `xception_deepfake.pth` | 80 MB | Xception (FaceForensics++) | Rössler et al., *FaceForensics++*, ICCV 2019 |
| `efficientnet_b4.pth` | 135 MB | SBI EfficientNet-B4 | Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images*, CVPR 2022 |
| `cnndetect_resnet50.pth` | 270 MB | CNNDetect ResNet-50 | Wang et al., *CNN-generated images are surprisingly easy to spot*, CVPR 2020 |

The CLIP-ViT-L/14 backbone (~890 MB) is automatically downloaded from HuggingFace (`openai/clip-vit-large-patch14`) on first run.

---

## 📚 References

```
@inproceedings{ojha2023universal,
  title={Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{shiohara2022sbi,
  title={Detecting Deepfakes with Self-Blended Images},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{wang2020cnndetect,
  title={CNN-generated images are surprisingly easy to spot... for now},
  author={Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A},
  booktitle={CVPR},
  year={2020}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🔬 forensic rigor and ⚡ engineering precision.
</p>
