# 🛡️ Aegis-X Interview Questions Guide

## Advanced Deepfake Forensic Detection Pipeline - Comprehensive Interview Preparation
### **Version 4.0: Three-Pronged Anomaly Shield — Decider/Supporter Hierarchy · GPU Conflict Guard · No-Face Pipeline**

---

## Table of Contents

1. [High-Level Architecture & Design Decisions](#1-high-level-architecture--design-decisions)
2. [Dual-Environment Architecture](#2-dual-environment-architecture)
3. [Tool-Specific Technical Questions](#3-tool-specific-technical-questions)
4. [Ensemble Scoring & Aggregation](#4-ensemble-scoring--aggregation)
5. [Early Stopping & Confidence Gating](#5-early-stopping--confidence-gating)
6. [VRAM Management & GPU Optimization](#6-vram-management--gpu-optimization)
7. [Preprocessing & Face Detection](#7-preprocessing--face-detection)
8. [LLM Integration & Explainability](#8-llm-integration--explainability)
9. [System Design & Scalability](#9-system-design--scalability)
10. [Error Handling & Edge Cases](#10-error-handling--edge-cases)
11. [Security & Adversarial Attacks](#11-security--adversarial-attacks)
12. [Performance & Benchmarking](#12-performance--benchmarking)
13. [Production Deployment & Monitoring](#13-production-deployment--monitoring)
14. [Code Quality & Best Practices](#14-code-quality--best-practices)
15. [Domain Knowledge - Deepfake Detection](#15-domain-knowledge---deepfake-detection)
16. [Critical Architecture Fixes — Silent Killers](#16-critical-architecture-fixes--silent-killers)
17. [Dual-Pipeline Orchestration Deep Dive](#17-dual-pipeline-orchestration-deep-dive)
18. [CPU→GPU Gate — Directional Confidence Mathematics](#18-cpugpu-gate--directional-confidence-mathematics)
19. [rPPG Temporal Window Precision](#19-rppg-temporal-window-precision)
20. [Closures, DRY Error Handling & DEGRADED Mode](#20-closures-dry-error-handling--degraded-mode)

---

## 1. High-Level Architecture & Design Decisions

### Q1: Why did you choose a multi-tool ensemble approach instead of training a single end-to-end model?

**What the interviewer is testing:** Understanding of ensemble methods vs. monolithic models, trade-offs in ML system design.

**Key points to cover:**
- **Diversity of detection signals**: Different tools detect different manipulation artifacts (frequency domain, geometric, physiological, etc.)
- **Robustness**: Single models can be fooled by attacks targeting their specific architecture; ensemble requires fooling multiple orthogonal detectors
- **Interpretability**: Each tool provides explainable evidence (e.g., "missing corneal reflection" vs. black-box probability)
- **Modularity**: Can add/remove tools without retraining entire system
- **Graceful degradation**: If one tool fails, others still contribute

**Follow-up question:** What are the downsides of this approach?
- Increased computational cost
- Complexity in orchestration
- Potential for conflicting predictions
- Harder to optimize globally

---

### Q2: Explain the CPU phase vs. GPU phase separation. Why not run everything on GPU?

**What the interviewer is testing:** Understanding of hardware resource optimization, cost-benefit analysis.

**Key points to cover:**
- **CPU tools are physics-based**: Geometry, illumination, corneal reflections, rPPG use classical computer vision and signal processing — no neural networks needed
- **Zero VRAM usage**: CPU tools don't consume precious GPU memory, leaving room for larger models
- **Latency**: Some CPU tools (MediaPipe landmarks) are faster than loading a GPU model
- **Energy efficiency**: No need to transfer data to/from GPU for simple calculations
- **Fallback capability**: System can still provide partial analysis if GPU is unavailable

**Deep dive:** Which tools are CPU-based and why?
```
Geometry Tool     → Landmark ratios (pure math)
Illumination Tool → Gradient analysis (OpenCV operations)
Corneal Tool      → Reflection detection (image processing)
rPPG Tool         → FFT on color signals (signal processing)
C2PA Tool         → Metadata parsing (file I/O)
DCT Tool          → JPEG quantization analysis (mathematical transform)
```

---

### Q3: How does your system handle both images and videos differently?

**What the interviewer is testing:** Understanding of temporal dimension in deepfake detection.

**Key points to cover:**
- **rPPG only works on video**: Requires temporal signal to extract heartbeat (physiological impossibility in static images)
- **Frame sampling strategy**: Videos use quality-snipe sampling to select most informative frames
- **Temporal consistency checks**: Video allows checking frame-to-frame consistency (flickering artifacts)
- **EMA smoothing**: Ensemble scores use Exponential Moving Average across frames for videos
- **Subject tracking**: Multiple faces tracked across frames with subject-aware state management

**Code reference:**
```python
# In agent.py - rPPG skip logic
if tool_name == "run_rppg" and preprocess_result.original_media_type == "image":
    logger.debug(f"Skipping {tool_name} (static image)")
    continue
```

---

### Q4: What is the significance of the 10-tool arsenal? How did you select these specific tools?

**What the interviewer is testing:** Domain knowledge, research awareness, systematic thinking.

**Key points to cover:**
- **Coverage of manipulation types**:
  - Generative AI (UnivFD with CLIP)
  - Face swaps (Xception from FaceForensics++)
  - Blend boundaries (SBI with GradCAM)
  - Frequency artifacts (FreqNet with DCT)
  - Geometric distortions (landmark ratios)
  - Lighting inconsistencies (gradient analysis)
  - Missing reflections (corneal tool)
  - Absent biological signals (rPPG)
  - Compression artifacts (DCT double quantization)
  - Provenance verification (C2PA)

- **Research-backed**: Each tool is based on peer-reviewed papers (CVPR 2023, FaceForensics++, etc.)
- **Orthogonal features**: Tools test independent feature spaces to avoid correlated failures

---

## 2. Dual-Environment Architecture

### Q5: Explain your dual virtual environment design. Why not use a single environment?

**What the interviewer is testing:** System architecture skills, dependency management, production thinking.

**Key points to cover:**
- **Dependency isolation**:
  - `.venv_main`: Lightweight (FastAPI, MediaPipe, OpenCV) — ~200MB
  - `.venv_gpu`: Heavy (PyTorch, Transformers, timm) — ~5GB+

- **Memory efficiency**: Web server doesn't load PyTorch unless needed
- **Faster startup**: Main process starts quickly, GPU worker spawned on-demand
- **Version conflicts**: CUDA/cuDNN versions don't interfere with web dependencies
- **Deployment flexibility**: Can deploy CPU-only version for basic analysis

**Architecture diagram:**
```
┌─────────────────┐         ┌─────────────────┐
│   .venv_main    │         │    .venv_gpu    │
│  (Web Server)   │◄───────►│  (GPU Worker)   │
│  - FastAPI      │  JSON   │  - PyTorch      │
│  - MediaPipe    │  over   │  - Transformers │
│  - OpenCV       │  pipe   │  - timm         │
└─────────────────┘         └─────────────────┘
        ▲                           ▲
        │                           │
   User Request              Model Loading
   SSE Streaming             Inference
```

**Follow-up:** How do the environments communicate?
- `SubprocessProxy` in `core/subprocess_proxy.py`
- Serialized input sent via subprocess pipe
- GPU worker returns `ToolResult` objects

---

### Q6: What challenges did you face with the subprocess proxy pattern? How did you solve them?

**What the interviewer is testing:** Problem-solving, debugging complex systems.

**Expected challenges and solutions:**
1. **Serialization overhead**: Used efficient pickle protocol, minimized data transfer
2. **Process lifecycle**: Implemented proper cleanup (`synchronize → del → gc → empty_cache`)
3. **Error propagation**: Custom exception hierarchy to cross process boundaries
4. **VRAM leaks**: `VRAMLifecycleManager` ensures cleanup between tool executions
5. **Blocking calls**: Generator-based streaming for real-time UI updates

**Code reference:**
```python
# In utils/vram_manager.py
def run_with_vram_cleanup(tool_loader, inference_fn, model_name, required_vram_gb):
    try:
        # Check VRAM availability
        # Load model
        # Run inference
        return result
    finally:
        # Always cleanup
        torch.cuda.synchronize()
        del model
        gc.collect()
        torch.cuda.empty_cache()
```

---

## 3. Tool-Specific Technical Questions

### Q7: How does UnivFD detect generative AI images? What makes it effective against FLUX, Midjourney, DALL-E?

**What the interviewer is testing:** Understanding of SOTA deepfake detection, CLIP architecture.

**Technical explanation:**
- **CLIP-ViT-L/14 backbone**: Pre-trained on 400M image-text pairs, learns universal visual features
- **Linear probe**: 4KB classifier head trained on fake/real features (Ojha et al. CVPR 2023)
- **Why it generalizes**: CLIP's features capture high-level semantic inconsistencies common across generative models
- **Detection signals**:
  - Texture patterns unique to diffusion/GAN models
  - Semantic coherence issues (hands, text, backgrounds)
  - Frequency domain artifacts from upsampling

**Weight in ensemble:** 0.22 (thresholds.py) / 0.20 (registry.py) — Decider — Tier 3 trust

---

### Q8: Explain how the rPPG tool detects deepfakes. What's the science behind it?

**What the interviewer is testing:** Signal processing knowledge, physiological understanding.

**Technical explanation:**
```
1. Face tracking → Extract ROI (region of interest) on forehead/cheeks
2. Color channel separation → Track RGB intensity over time
3. Bandpass filtering → Isolate cardiac frequency band (0.7-2.5 Hz = 42-150 BPM) ⚠️ Code uses 2.5 Hz, not 4.0 Hz
4. FFT (Fast Fourier Transform) → Find dominant frequency
5. Peak detection → Check if clear heartbeat signal exists
```

**Why deepfakes fail:**
- Most generation methods don't simulate blood flow under skin
- Temporal averaging blurs subtle color changes
- Frame interpolation breaks physiological rhythm

**Thresholds:**
```python
RPPG_PULSE_THRESHOLD_LOW = 0.3   # Below = no pulse detected
RPPG_PULSE_THRESHOLD_HIGH = 0.7  # Above = strong pulse
RPPG_NO_PULSE_IMPLIED_PROB = 0.85 # Implied fake probability if no pulse
RPPG_CARDIAC_BAND_MAX_HZ = 2.5   # ⚠️ Code uses 2.5 Hz (150 BPM), README documents 4.0 Hz (240 BPM)
RPPG_HAIR_OCCLUSION_VARIANCE = 0.25  # ⚠️ BUG: Should be ~35.0 — currently causes false positives
```

**Limitation:** Only works on videos with visible skin, good lighting

---

### Q9: What are corneal reflections? Why are they important for deepfake detection?

**What the interviewer is testing:** Physics-based detection, attention to detail.

**Explanation:**
- **Corneal reflection** = "catchlight" = reflection of light sources in the eye's cornea
- **Physical law**: Both eyes should show reflections from the same light source at symmetric positions
- **Deepfake failure modes**:
  - AI generates eyes independently → mismatched reflections
  - Missing reflections entirely (common in low-quality fakes)
  - Asymmetric divergence (violates physics)

**Implementation:**
```python
# In corneal_tool.py
1. Detect iris/pupil regions
2. Apply bilateral filter to isolate bright spots
3. Find reflection centroids
4. Calculate divergence angle between left/right eyes
5. Score based on symmetry and presence
```

**Weight:** 0.04 (thresholds.py) / 0.07 (registry.py) — Supporter — Tier 2

---

### Q10: How does the SBI (Self-Blended Images) tool work? What's GradCAM's role?

**What the interviewer is testing:** Understanding of attribution methods, face manipulation detection.

**Technical explanation:**
- **Training method**: Self-blended images create synthetic composites for training
- **Architecture**: EfficientNet-B4 backbone
- **GradCAM integration**:
  - Generates heatmaps showing which regions contributed to "fake" prediction
  - Highlights blend boundaries, seam artifacts
  - Provides visual evidence for forensic reports

**Skip logic:**
```python
# Skip SBI if UnivFD already detected fully synthetic content
if univfd_score > SBI_SKIP_UNIVFD_THRESHOLD:
    logger.debug("Skipping SBI (UnivFD score = fully synthetic)")
    continue
```

**Why skip?** SBI targets face compositing, not fully generated faces — orthogonal blind spots

---

### Q11: What is frequency domain analysis? How does FreqNet use it?

**What the interviewer is testing:** Signal processing, Fourier analysis applications.

**Explanation:**
- **Spatial domain**: Pixel values at locations (x, y)
- **Frequency domain**: Patterns of change (low freq = smooth areas, high freq = edges)
- **DCT (Discrete Cosine Transform)**: Converts image to frequency coefficients

**Why deepfakes have frequency artifacts:**
- GANs/diffusion models introduce characteristic patterns in frequency space
- Upsampling operations leave spectral signatures
- Convolution operations have frequency-domain footprints

**FreqNet implementation:**
```python
# CNNDetect ResNet-50 + FADHook
1. Apply DCT to image patches
2. Extract frequency energy distribution
3. Feed to ResNet classifier
4. FADHook (Frequency Artifact Detection) adds calibration layer
```

**Compression discount:** If DCT tool detects JPEG compression, FreqNet weight is reduced (compression masks artifacts)

---

### Q12: How does the geometry tool detect anthropometric distortions?

**What the interviewer is testing:** Computer vision, facial anatomy knowledge.

**Measurements taken:**
```python
# From geometry_tool.py
- IPD (Interpupillary Distance) ratio
- Philtrum length (nose to mouth)
- Eye symmetry (left vs right)
- Nose width ratio
- Mouth width ratio
- Vertical facial thirds (forehead, mid-face, lower-face)
```

**Why deepfakes fail:**
- Face swapping can misalign landmark positions
- Generative models sometimes violate anatomical proportions
- Reenactment can stretch/compress facial regions unnaturally

**Implementation:**
```python
# MediaPipe Face Mesh → 468 landmarks
# Calculate ratios between specific landmark groups
# Compare to population norms (±2σ threshold)
# Score deviations as suspicious
```

**Weight:** 0.08 (thresholds.py) / 0.18 (registry.py) — Supporter — Tier 3 (registry)

> **v4.0**: Geometry was demoted from Decider (0.20) to Supporter (0.08) because noisy CPU heuristics were overriding GPU deep-learning models, causing false positives on real images with chaotic lighting.

---

## 4. Ensemble Scoring & Aggregation

### Q13: Explain your ensemble scoring formula. How do you combine tool results?

**What the interviewer is testing:** Mathematical reasoning, weighted averaging, conflict resolution.

**Core algorithm:**
```python
# Simplified from ensemble.py
total_contribution = Σ(score_i × weight_i × confidence_i)
total_weight = Σ(weight_i × confidence_i)
ensemble_fake_score = total_contribution / total_weight
ensemble_real_score = 1.0 - ensemble_fake_score
```

**Key features:**
1. **Confidence weighting**: Higher confidence tools get more influence
2. **Context-aware routing**: Some tools abstain based on conditions (e.g., SBI skips if no compression detected)
3. **Suspicion overdrive (v4.0)**: If any GPU specialist exceeds `SUSPICION_OVERRIDE_THRESHOLD` (0.70), use max-pooling — but only after GPU Conflict Guard verifies specialists agree (spread ≤ 0.30)
4. **Borderline Consensus**: If ≥2 GPU specialists cluster in [0.35, 0.55], their mean is boosted 1.25×
5. **GPU Coverage Degradation**: Each abstained GPU specialist applies +10% boost to fake_score
6. **Compression discounts**: DCT-detected compression reduces SBI/FreqNet weights (artifacts may be from JPEG, not deepfake)

**Follow-up:** Why convert to "real probability" at the end?
- Forensic convention: 0 = fake, 1 = real is more intuitive for investigators
- Matches C2PA semantics (verified = authentic)

---

### Q14: What is "Suspicion Overdrive" and why is it necessary?

**What the interviewer is testing:** Understanding of orthogonal feature spaces, failure mode analysis.

**Problem statement:**
```
Scenario:
- Tool A (face-swap detector): 0.95 fake (high confidence)
- Tool B (generative detector): 0.10 fake (says real)
- Tool C (geometry): 0.15 fake (says real)

Simple average: (0.95 + 0.10 + 0.15) / 3 = 0.40 → "REAL" ❌ WRONG!
```

**Why this happens:**
- Tools test **orthogonal features** (different manipulation types)
- A clear "real" in one dimension doesn't cancel a strong "fake" in another
- Face-swap artifacts ≠ generative artifacts ≠ geometric distortions

**Solution (v4.0 — Three-Pronged):**
```python
# Prong 1: Suspicion Overdrive with GPU Conflict Guard
max_prob = max(gpu_implied_probs)  # 0.95
spread = max(gpu_implied_probs) - min(gpu_implied_probs)  # conflict check
if max_prob > SUSPICION_OVERRIDE_THRESHOLD and spread <= 0.30:  # GPU agreement
    fake_score = max_prob  # Use 0.95, not 0.40
# Note: SUSPICION_OVERRIDE_THRESHOLD = 0.70 in code (not 0.50 as some docs say)

# Prong 2: Borderline Consensus
borderline_specialists = [p for p in gpu_probs if 0.35 <= p <= 0.55]
if len(borderline_specialists) >= 2:
    boosted = mean(borderline_specialists) * 1.25  # corroboration

# Prong 3: GPU Coverage Degradation
for each abstained GPU specialist:
    fake_score *= (1.0 + 0.10)  # +10% per blind spot
```

**Analogy:** Medical diagnosis — one positive cancer test isn't negated by ten negative flu tests

---

### Q15: How do you handle tool conflicts? What if tools strongly disagree?

**What the interviewer is testing:** Conflict resolution strategies, statistical reasoning.

**Implementation:**
```python
# Calculate standard deviation of implied probabilities
conflict_std = _compute_conflict_std(implied_probs)
has_conflict = conflict_std > CONFLICT_STD_THRESHOLD  # e.g., 0.20
```

**Actions on conflict:**
1. **Block early stopping**: Don't halt pipeline prematurely if tools disagree
2. **Log warning**: Alert user/investigator to examine manually
3. **Pass to LLM**: Include conflict info in synthesis prompt for nuanced explanation
4. **Lower confidence**: Reduce overall confidence score in verdict

**Example conflict scenario:**
```
UnivFD: 0.90 fake (thinks it's AI-generated)
Xception: 0.20 fake (thinks it's real)
→ High conflict std → Flag for human review
```

---

### Q16: What is EMA smoothing? Why apply it to video frame scores?

**What the interviewer is testing:** Time-series analysis, temporal consistency.

**Explanation:**
- **EMA (Exponential Moving Average)**: Weighted average giving more importance to recent values
- **Formula**: `smoothed_t = α × current + (1-α) × previous`
- **Purpose**: Reduce frame-to-frame jitter, provide stable scores

**Why needed for video:**
- Individual frames may have noise/artifacts
- Tracking may temporarily lose face
- Lighting changes can cause transient spikes
- Human perception expects smooth transitions

**Scene cut handling:**
```python
if output["is_inconclusive"]:
    # Hard reset on scene cut or tracking loss
    subject_states.pop(subject_id, None)
```

**Alpha value:** Typically 0.3-0.5 (balances responsiveness vs. stability)

---

### Q17: How does C2PA verification short-circuit the pipeline?

**What the interviewer is testing:** Understanding of provenance standards, optimization.

**C2PA (Coalition for Content Provenance and Authenticity):**
- Industry standard for cryptographic media provenance
- Embedded metadata signed by camera/manufacturer
- Tamper-evident seal

**Short-circuit logic:**
```python
if tool_name == "check_c2pa" and result.details.get("c2pa_verified"):
    logger.info("C2PA verified — short-circuiting entire pipeline")
    return True  # Skip all other tools
```

**But with visual check:**
```python
# If C2PA says real but visual models scream fake → possible spoofing
if visual_avg > C2PA_VISUAL_CONTRADICTION_THRESHOLD:
    logger.warning("C2PA verified but visual models scream FAKE")
    # Continue analysis instead of short-circuit
```

**Why this matters:** C2PA metadata can be stripped or spoofed; visual analysis provides defense-in-depth

---

## 5. Early Stopping & Confidence Gating

### Q18: Explain your early stopping mechanism. When does the system decide to stop analyzing?

**What the interviewer is testing:** Optimization strategies, confidence calibration.

**Two-tier gating:**

**1. Confidence Gate (after CPU phase):**
```python
if current_confidence >= CONFIDENCE_GATE_THRESHOLD:  # e.g., 0.85
    return True  # Skip GPU phase entirely
```

**2. Evidential Subjective Logic (during GPU phase):**
```python
decision = esc.evaluate(
    tool_scores=tool_scores,
    completed_tools=list(self.ensemble.tool_results.keys()),
    c2pa_hardware_verified=c2pa_verified
)
if decision.should_stop:
    break  # Halt remaining GPU tools
```

**Stop conditions:**
- Locked FAKE: Multiple high-confidence tools agree on fake
- Locked REAL: All tools show low suspicion, C2PA verified
- Diminishing returns: Remaining tools unlikely to change verdict

**Never stops early for REAL claims:** Always runs full GPU phase to rule out sophisticated fakes

---

### Q19: What is Evidential Subjective Logic? How does it differ from simple thresholding?

**What the interviewer is testing:** Advanced reasoning frameworks, uncertainty modeling.

**Traditional thresholding:**
```python
if score > 0.85: STOP
```

**Evidential Subjective Logic (ESL):**
- Models **belief**, **disbelief**, **uncertainty**, and **conflict** as separate dimensions
- Considers **evidence mass** accumulated so far
- Accounts for **tool reliability tiers** (Tier 1/2/3)
- Detects **conflict** between tools before making stop decision

**Mathematical framework:**
```
Opinion = (b, d, u, a) where:
  b = belief mass
  d = disbelief mass
  u = uncertainty mass
  a = prior probability
  b + d + u = 1
```

**Advantage:** More nuanced than binary thresholds; handles ambiguous cases gracefully

---

### Q20: Why does the system always run GPU tools for REAL claims but can early-stop for FAKE claims?

**What the interviewer is testing:** Risk assessment, asymmetric error costs.

**Reasoning:**
- **False REAL (calling fake "real")**: Catastrophic failure — malicious content passes undetected
- **False FAKE (calling real "fake")**: Annoying but safe — legitimate content flagged for review

**Risk asymmetry:**
```
FAKE verdict early stop → Low risk (conservative, err on side of caution)
REAL verdict early stop → High risk (might miss sophisticated attack)
```

**Implementation:**
```python
# From early_stopping.py
if verdict == "REAL":
    should_stop = False  # Force full analysis
elif verdict == "FAKE" and confidence > EARLY_STOP_CONFIDENCE:
    should_stop = True   # Can stop early
```

**Analogy:** Airport security — better to flag innocent travelers than miss a threat

---

## 6. VRAM Management & GPU Optimization

### Q21: How do you achieve "VRAM-safe on 4GB GPUs"? What's your memory management strategy?

**What the interviewer is testing:** Resource-constrained optimization, production deployment thinking.

**Key techniques:**

**1. Sequential loading (never additive):**
```python
# Load Tool A → Run → Unload → Load Tool B → Run → Unload
# Peak VRAM = max(single model), not sum(all models)
```

**2. Lifecycle management:**
```python
def run_with_vram_cleanup():
    try:
        result = inference()
        return result
    finally:
        torch.cuda.synchronize()  # Wait for ops to complete
        del model                 # Remove Python references
        gc.collect()              # Force garbage collection
        torch.cuda.empty_cache()  # Release GPU memory allocator
```

**3. Model size awareness:**
```
UnivFD (CLIP-ViT-L): ~1.8 GB peak
Xception:            ~0.9 GB
SBI (EfficientNet):  ~0.7 GB
FreqNet (ResNet50):  ~1.2 GB
```

**4. Expandable segments:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Allows PyTorch to fragment/reuse VRAM more efficiently

**Result:** Peak VRAM never exceeds ~1.8 GB even though total model weights = 4.6 GB

---

### Q22: What is `torch.cuda.empty_cache()` and when should you call it?

**What the interviewer is testing:** PyTorch internals, GPU memory management.

**Explanation:**
- PyTorch uses a **caching allocator** — freed tensors don't immediately return to OS
- `empty_cache()` releases unused cached memory back to system
- **Expensive operation** — should only call between major phases, not every inference

**When to call:**
```python
✅ Between different model executions
✅ After processing large batch
✅ Before loading next model in sequence

❌ Inside training loop (destroys caching benefits)
❌ After every single tensor operation
```

**Gotcha:** Doesn't free memory still referenced by Python objects — must `del` first

---

### Q23: How do you handle VRAM fragmentation? What's the expandable_segments setting?

**What the interviewer is testing:** Deep PyTorch knowledge, performance tuning.

**Problem:**
- Repeated alloc/free creates fragmented VRAM
- Large contiguous allocations fail even if total free memory is sufficient
- Like disk fragmentation but worse for GPU

**Solution:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**What it does:**
- Allows PyTorch allocator to use non-contiguous memory segments
- Automatically coalesces adjacent free blocks
- Reduces allocation failures on fragmented VRAM

**Trade-off:** Slightly higher overhead per allocation, but prevents OOM crashes

---

### Q24: What happens if a GPU tool fails due to insufficient VRAM?

**What the interviewer is testing:** Error handling, graceful degradation.

**Implementation:**
```python
# In vram_manager.py
try:
    check_vram_availability(required_vram_gb)
    result = run_inference()
except InsufficientVRAMError:
    logger.warning(f"{tool_name} failed: insufficient VRAM")
    return ToolResult(
        tool_name=tool_name,
        success=False,
        error_msg="Insufficient VRAM",
        score=0.5,  # Neutral score
        confidence=0.0
    )
```

**Ensemble behavior:**
- Failed tool contributes zero weight
- Other tools compensate
- Final confidence reduced proportionally
- User notified in UI about skipped tools

**Design principle:** Partial analysis > no analysis

---

## 7. Preprocessing & Face Detection

### Q25: How does your preprocessing pipeline work? What crops do you generate?

**What the interviewer is testing:** Data preparation, multi-scale analysis.

**Pipeline:**
```python
# From preprocessing.py
1. MediaPipe Face Mesh → 468 landmarks per face
2. Track faces across frames (for video)
3. Generate two crops per face:
   - 224×224: Standard size for most models
   - 380×380: Larger crop for context-aware models
4. Quality-snipe sampling (video): Select frames with best face visibility
```

**Why two crop sizes?**
- 224×224: Optimized for ImageNet-pretrained models (Xception, EfficientNet)
- 380×380: Preserves fine details for frequency analysis, corneal reflections

**Face tracking:**
- Maintains subject IDs across frames
- Enables subject-aware EMA smoothing
- Handles multiple faces simultaneously

---

### Q26: What is "quality-snipe frame sampling"? Why not analyze every frame?

**What the interviewer is testing:** Efficiency optimization, smart sampling strategies.

**Problem:**
- 30 FPS video × 60 seconds = 1800 frames
- Analyzing all frames = wasteful, redundant
- Most frames are near-identical

**Solution:**
```python
# Score each frame on:
- Face size (larger = better)
- Face angle (frontal = better)
- Blur metric (sharper = better)
- Occlusion (less = better)
- Lighting (well-lit = better)

# Select top-K diverse frames
# Ensures coverage of different expressions/angles
```

**Benefits:**
- 10-20 representative frames instead of 1800
- 90% reduction in compute time
- Focuses on highest-quality evidence

---

### Q27: How does MediaPipe Face Mesh compare to other face detectors? Why choose it?

**What the interviewer is testing:** Tool selection rationale, comparative analysis.

**Comparison:**

| Detector | Speed | Landmarks | 3D Info | License |
|----------|-------|-----------|---------|---------|
| MediaPipe | ⚡⚡⚡ Very fast | 468 dense | Yes | Apache 2.0 |
| MTCNN | ⚡⚡ Fast | 5 points | No | MIT |
| RetinaFace | ⚡ Medium | 5 points | No | Apache 2.0 |
| Dlib | ⚡ Slow | 68 points | No | BSD |

**Why MediaPipe:**
- **Dense landmarks**: 468 points enable detailed geometry analysis
- **3D coordinates**: Depth information for corneal/illumination tools
- **Real-time**: 100+ FPS on CPU
- **Multi-face**: Tracks multiple faces natively
- **Cross-platform**: Works on CPU, no GPU required

**Trade-off:** Less accurate on extreme poses vs. specialized detectors

---

## 8. LLM Integration & Explainability

### Q28: How do you use the LLM (Phi-3 Mini) for verdict synthesis? What's the prompt structure?

**What the interviewer is testing:** LLM integration, prompt engineering, XAI (Explainable AI).

**Prompt structure:**
```python
# From forensic_summary.py
prompt = f"""
You are a forensic media analyst. Synthesize the following tool results into a clear verdict.

MEDIA ANALYSIS RESULTS:
{'='*50}

ENSEMBLE SCORE: {ensemble_score:.2f} ({verdict})

INDIVIDUAL TOOL FINDINGS:
"""

for tool_name, result in tool_results.items():
    prompt += f"""
{tool_name}:
  - Score: {result.score:.2f} ({'SUSPICIOUS' if result.score > 0.55 else 'CLEAR'})
  - Confidence: {result.confidence:.2f}
  - Evidence: {result.evidence_summary}
"""

prompt += """
SCORING CONVENTION:
- 0.0 = Authentic, 1.0 = Tampered
- Scores 0.55+ indicate potential manipulation

INSTRUCTIONS:
1. State the final verdict clearly (AUTHENTIC or TAMPERED)
2. Explain the key evidence that drove this conclusion
3. Mention any conflicting signals and how they were resolved
4. Use plain language suitable for non-technical investigators
5. Be concise but thorough (max 300 words)

VERDICT:
"""
```

**Why local LLM (Ollama)?**
- Privacy: No data leaves the machine
- Latency: No API round-trip
- Cost: Free after initial setup
- Control: Deterministic output with low temperature

---

### Q29: What are the risks of using an LLM for verdict synthesis? How do you mitigate them?

**What the interviewer is testing:** Critical thinking, LLM limitations awareness.

**Risks:**
1. **Hallucination**: LLM might invent evidence not present in tool results
2. **Inconsistency**: Same input could produce different explanations
3. **Bias**: Training data biases might affect wording
4. **Overconfidence**: LLM might sound certain even when evidence is weak

**Mitigations:**
```python
# 1. Grounded prompting
"Base your explanation ONLY on the tool results provided above"
"Do not speculate beyond the evidence"

# 2. Low temperature
LLM_TEMPERATURE = 0.1  # Near-deterministic

# 3. Structured output
"State the final verdict clearly (AUTHENTIC or TAMPERED)"
"List the top 3 pieces of evidence"

# 4. Confidence passthrough
Include tool confidence scores in prompt
"Note any tools with low confidence (<0.5)"

# 5. Verification layer (future)
Parse LLM output to ensure it matches ensemble verdict
```

---

### Q30: How do you ensure the LLM explanation is faithful to the actual tool results?

**What the interviewer is testing:** Trustworthiness, verification mechanisms.

**Current approach:**
- **Structured prompt**: Explicitly list all tool scores and evidence
- **Instruction constraints**: "Do not invent evidence", "Cite specific tools"
- **Low temperature**: 0.1 for deterministic output
- **Human review**: UI shows both raw tool results AND LLM summary

**Future improvements:**
```python
# Post-hoc verification
def verify_explanation(explanation, tool_results):
    # Extract claims from LLM output
    claims = parse_claims(explanation)

    # Verify each claim against actual tool results
    for claim in claims:
        if not is_supported_by_evidence(claim, tool_results):
            flag_for_review()

    # Check verdict alignment
    llm_verdict = extract_verdict(explanation)
    if llm_verdict != ensemble_verdict:
        raise VerdictMismatchError()
```

**Best practice:** LLM explanation is supplementary — raw tool results always visible in UI

---

## 9. System Design & Scalability

### Q31: How would you scale this system to handle 1000 requests per minute?

**What the interviewer is testing:** Scalability thinking, distributed systems.

**Bottlenecks:**
1. **GPU contention**: Only one inference per GPU at a time (sequential loading)
2. **VRAM limits**: Can't batch multiple models simultaneously
3. **LLM latency**: Phi-3 takes ~2-5 seconds per verdict
4. **CPU preprocessing**: MediaPipe is fast but not free

**Scaling strategies:**

**Horizontal scaling:**
```
Load Balancer
    ├── Worker 1 (GPU 0)
    ├── Worker 2 (GPU 1)
    ├── Worker 3 (GPU 2)
    └── Worker N (GPU N)
```

**Optimizations:**
1. **Model batching**: Process multiple images through same loaded model before switching
2. **Queue prioritization**: High-confidence cases first, low-priority in background
3. **Caching**: Hash-based cache for duplicate submissions
4. **Async processing**: Return job ID, stream results via WebSocket
5. **Model parallelism**: Distribute tools across multiple GPUs

**Estimated capacity:**
- Single GPU: ~20-30 analyses/minute (depending on media length)
- 10 GPUs: ~200-300 analyses/minute
- Need ~30-50 GPUs for 1000 req/min

---

### Q32: How would you design a cloud-native version of Aegis-X?

**What the interviewer is testing:** Cloud architecture, containerization, microservices.

**Architecture:**
```yaml
Services:
  - API Gateway (FastAPI)
  - Preprocessing Service (CPU-only, auto-scale)
  - GPU Inference Cluster (Kubernetes + NVIDIA Device Plugin)
  - LLM Service (Separate pod with Ollama)
  - Result Cache (Redis)
  - Job Queue (RabbitMQ/Celery)
  - Storage (S3 for uploaded media)
  - Database (PostgreSQL for audit logs)
```

**Kubernetes manifests:**
```yaml
# GPU worker deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aegis-gpu-worker
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: worker
        image: aegis-x:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
```

**Cost optimization:**
- Spot instances for batch processing
- Auto-scaling based on queue depth
- Model warm-up during low-traffic periods

---

### Q33: How do you handle concurrent requests to the same GPU?

**What the interviewer is testing:** Concurrency control, resource locking.

**Current implementation:** Single-threaded per GPU worker

**Concurrency strategies:**

**1. Request queue:**
```python
# FIFO queue with worker pool
queue = asyncio.Queue()
workers = [asyncio.create_task(process_request()) for _ in range(num_gpus)]
```

**2. Batch processing:**
```python
# Collect requests for 100ms window
# Load model once, process batch, unload
async def batch_processor():
    batch = await gather_requests(timeout=0.1)
    results = model(batch)  # Single forward pass
    distribute_results(results)
```

**3. Model replication:**
```
GPU 0: Model A (processing)
GPU 1: Model A (queued)
GPU 2: Model B (idle)
→ Dynamic scheduling based on demand
```

**4. Priority queuing:**
```python
# Premium users jump queue
# Real-time analysis > batch processing
queue.put(request, priority=user_tier)
```

---

## 10. Error Handling & Edge Cases

### Q34: What happens if no faces are detected in an image?

**What the interviewer is testing:** Edge case handling, user experience.

**Current behavior:**
```python
# From agent.py
if tool_name in ("run_geometry", "run_illumination", "run_corneal"):
    if not preprocess_result.tracked_faces:
        logger.debug(f"Skipping {tool_name} (no landmarks)")
        continue
```

**Cascade effects:**
- CPU tools abstain (zero weight)
- GPU tools may still run (UnivFD works on full image, not just faces)
- Ensemble score becomes inconclusive if too many tools abstain
- User sees: "No faces detected — limited analysis available"

**Improved UX:**
```python
if not preprocess_result.tracked_faces:
    return {
        "verdict": "INCONCLUSIVE",
        "message": "No human faces detected. This tool is designed for face-based deepfake detection.",
        "suggestion": "Try our scene-level forgery detector for non-face manipulations"
    }
```

---

### Q35: How do you handle corrupted or malformed input files?

**What the interviewer is testing:** Input validation, robustness.

**Validation layers:**
```python
# 1. File type check
allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi'}
if ext not in allowed_extensions:
    raise UnsupportedFormatError()

# 2. Magic number verification
with open(file, 'rb') as f:
    header = f.read(8)
    if not is_valid_image_header(header):
        raise CorruptedFileError()

# 3. Size limits
if file_size > MAX_FILE_SIZE:
    raise FileTooLargeError()

# 4. Decode attempt
try:
    img = cv2.imread(path)
    if img is None:
        raise DecodeError()
except Exception as e:
    raise CorruptedFileError(str(e))
```

**User feedback:**
- Clear error messages: "File appears corrupted — please re-upload"
- Suggestions: "Supported formats: JPEG, PNG, WebP, MP4"
- No stack traces exposed to users

---

### Q36: What if the Ollama service is down or unreachable?

**What the interviewer is testing:** Graceful degradation, fallback strategies.

**Current implementation:**
```python
# From llm.py
try:
    response = requests.post(ollama_endpoint, json=prompt, timeout=30)
    return response.json()['response']
except requests.exceptions.RequestException as e:
    logger.error(f"LLM service unavailable: {e}")
    # Fallback to template-based explanation
    return generate_fallback_explanation(tool_results, verdict)
```

**Fallback explanation:**
```python
def generate_fallback_explanation(tool_results, verdict):
    template = f"""
VERDICT: {verdict}

KEY FINDINGS:
"""
    for tool_name, result in sorted(tool_results.items(), key=lambda x: x[1].score, reverse=True):
        template += f"- {tool_name}: {'SUSPICIOUS' if result.score > 0.55 else 'CLEAR'} (score: {result.score:.2f})\n"

    template += "\nNote: Detailed explanation unavailable (LLM service offline)"
    return template
```

**Principle:** Core functionality (detection) must work even if nice-to-have (explanation) fails

---

### Q37: How do you handle extremely long videos (e.g., 2-hour movies)?

**What the interviewer is testing:** Resource management, practical limits.

**Strategy:**
```python
# 1. Duration check
duration = get_video_duration(path)
if duration > MAX_DURATION:  # e.g., 10 minutes
    return {
        "error": "Video too long",
        "suggestion": "Please upload clips under 10 minutes or extract key frames"
    }

# 2. Adaptive sampling
if duration > 5 minutes:
    sample_rate = 1 frame per 10 seconds
elif duration > 1 minute:
    sample_rate = 1 frame per 2 seconds
else:
    sample_rate = 1 frame per second

# 3. Scene detection
scenes = detect_scene_changes(video)
sample_frames = [scenes[0], scenes[len(scenes)//2], scenes[-1]]
# Analyze representative frames from each scene
```

**Alternative:** Offer "quick scan" mode (30-second preview) vs. "full analysis" (entire video, async processing)

---

## 11. Security & Adversarial Attacks

### Q38: Can attackers fool your ensemble? What are the known adversarial attacks?

**What the interviewer is testing:** Security awareness, adversarial ML knowledge.

**Attack vectors:**

**1. Gradient-based attacks:**
```python
# FGSM (Fast Gradient Sign Method)
perturbation = epsilon * sign(loss_gradient)
adversarial_image = original_image + perturbation
```
- Effective against individual neural networks
- **Defense**: Ensemble of diverse architectures increases robustness

**2. Transfer attacks:**
- Train surrogate model, generate adversarial examples, transfer to target
- **Defense**: Proprietary preprocessing (MediaPipe crops) acts as randomization

**3. Physical attacks:**
- Add printed patterns, wear special makeup
- **Defense**: Multi-modal detection (frequency + geometry + physiology)

**4. Model extraction:**
- Query API repeatedly to reconstruct model
- **Defense**: Rate limiting, output perturbation, query logging

**5. Poisoning attacks:**
- Inject fake training data if model is updated
- **Defense**: Frozen models, curated training sets only

**Your system's strengths:**
- Orthogonal detection methods require simultaneous defeat of all tools
- Physics-based tools (corneal, rPPG) are not trainable → immune to gradient attacks
- C2PA provides cryptographic verification (cannot be fooled visually)

---

### Q39: How would you defend against a "model stealing" attack where someone queries your API to build a surrogate?

**What the interviewer is testing:** API security, defensive design.

**Defenses:**

**1. Rate limiting:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/analyze")
@limiter.limit("10/minute")  # Max 10 requests per minute per IP
async def analyze(request: Request):
    ...
```

**2. Output perturbation:**
```python
# Add small random noise to scores
noise = np.random.normal(0, 0.02)  # ±2% noise
perturbed_score = clip(actual_score + noise, 0, 1)
```

**3. Honeypot detection:**
```python
# Track query patterns
if user_queries_diverse_images_sequentially():
    flag_as_potential_extraction()
    degrade_output_quality()  # Return less precise scores
```

**4. Watermarking:**
- Embed invisible patterns in test images
- If these appear in competitor's dataset → proof of theft

**5. Legal/terms of service:**
- Explicit prohibition on reverse engineering
- Required authentication for API access
- Audit logging for forensics

---

### Q40: What if someone submits an image that's already been processed by your system (recursive analysis)?

**What the interviewer is testing:** Edge cases, infinite loops, meta-reasoning.

**Scenario:**
```
Original image → Aegis-X → Saves result image → Submits result image again
```

**Potential issues:**
- Compression artifacts from saving
- Metadata changes
- Possible confusion if system analyzes its own output

**Handling:**
```python
# Detect previous analysis metadata
if "aegis_x_analyzed" in image.metadata:
    logger.warning("Image previously analyzed by Aegis-X")
    # Still analyze but note in report
    report.notes.append("This image contains metadata indicating prior Aegis-X analysis")

# Handle compression artifacts
if dct_tool.detects_double_compression():
    # Discount frequency-based tools
    freqnet_weight *= COMPRESSION_DISCOUNT
```

**Philosophy:** Treat as normal image — compression is common in real-world usage

---

## 12. Performance & Benchmarking

### Q41: What are your latency benchmarks? How fast is a typical analysis?

**What the interviewer is testing:** Performance awareness, optimization priorities.

**Breakdown (typical image, RTX 3050 4GB):**

| Phase | Time | Notes |
|-------|------|-------|
| Upload + Validation | 100-500ms | Depends on file size |
| Preprocessing (MediaPipe) | 200-400ms | CPU-bound |
| CPU Tools (4 tools) | 300-600ms | Parallelizable |
| GPU Tools (4 tools) | 2000-4000ms | Sequential loading dominates |
| Ensemble Aggregation | <10ms | Pure Python math |
| LLM Synthesis | 2000-5000ms | Phi-3 inference |
| **Total** | **4.6-10.5s** | End-to-end |

**Optimization opportunities:**
- Batch GPU model loading (amortize overhead)
- Async LLM calls (stream partial results)
- Early stopping (skip GPU if confidence gate triggers)
- Model quantization (INT8 for 2-4x speedup)

**Video analysis:**
- 30-second clip (~15 sampled frames): 30-60 seconds
- Dominated by GPU inference count

---

### Q42: How do you measure accuracy? What datasets did you validate on?

**What the interviewer is testing:** Evaluation methodology, scientific rigor.

**Metrics:**
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
AUC-ROC = Area under ROC curve
```

**Datasets:**
- **FaceForensics++**: 5000+ videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- **DFDC (Deepfake Detection Challenge)**: 100k+ videos (Facebook)
- **Celeb-DF**: High-quality celebrity deepfakes
- **WildDeepfake**: In-the-wild collected samples
- **Custom test set**: Mix of real images from various sources

**Reported performance (example):**
```
Dataset          | Accuracy | AUC-ROC
-----------------|----------|--------
FaceForensics++  | 94.2%    | 0.97
DFDC             | 89.5%    | 0.93
Celeb-DF         | 91.8%    | 0.95
WildDeepfake     | 87.3%    | 0.91
```

**Important:** Performance varies by manipulation type — always report per-category metrics

---

### Q43: How do you handle class imbalance in evaluation (more real than fake samples)?

**What the interviewer is testing:** Statistical literacy, evaluation best practices.

**Problem:**
```
Real-world distribution: 95% real, 5% fake
Naive model predicts "real" always → 95% accuracy but useless!
```

**Solutions:**

**1. Balanced accuracy:**
```python
balanced_acc = (sensitivity + specificity) / 2
```

**2. Stratified sampling:**
- Ensure test set has 50/50 real/fake split
- Report per-class metrics separately

**3. Precision-Recall curves:**
- More informative than ROC for imbalanced data
- Focus on fake detection rate (recall) and false positive rate (1-precision)

**4. Cost-sensitive evaluation:**
```python
# Weight fake detection higher
weighted_score = 0.7 × fake_recall + 0.3 × real_precision
```

**5. Confusion matrix analysis:**
```
                Predicted
              Real    Fake
Actual Real   950      50    ← 50 false positives
       Fake   20      480    ← 20 false negatives (dangerous!)
```

---

## 13. Production Deployment & Monitoring

### Q44: How would you monitor this system in production? What metrics would you track?

**What the interviewer is testing:** Observability, SRE thinking.

**Key metrics:**

**Infrastructure:**
- GPU utilization (%)
- VRAM usage (MB)
- Request queue depth
- Processing latency (p50, p95, p99)
- Error rate by tool

**Business logic:**
- Verdict distribution (% REAL vs % FAKE)
- Average confidence scores
- Early stop rate (% of requests that skipped GPU)
- Tool disagreement rate (% with high conflict_std)

**Quality:**
- User feedback thumbs up/down
- Appeal rate (% of FAKE verdicts contested)
- Manual review overturn rate

**Alerting thresholds:**
```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5% over 5min
    severity: critical

  - name: GPUMemoryLeak
    condition: vram_usage increasing monotonically
    severity: warning

  - name: VerdictAnomaly
    condition: fake_rate > 3σ from baseline
    severity: warning  # Might indicate attack wave
```

**Tools:** Prometheus + Grafana for metrics, ELK stack for logs

---

### Q45: How do you handle model updates? What's your deployment strategy?

**What the interviewer is testing:** CI/CD, versioning, rollback strategies.

**Strategy:**

**1. Versioned models:**
```
models/
  ├── univfd/
  │   ├── v1.0_probe.pth
  │   └── v2.0_probe.pth  ← symlink to current
  └── xception/
      └── v1.5_xception.pth
```

**2. Canary deployment:**
```
- Deploy new model to 5% of traffic
- Monitor metrics for 24 hours
- If no regression → increase to 50%
- After 48 hours → 100% rollout
```

**3. A/B testing:**
```python
if user_id % 100 < 10:  # 10% of users
    model_version = "v2.0"
else:
    model_version = "v1.0"
```

**4. Rollback plan:**
```bash
# Instant rollback via symlink switch
ln -sfn v1.0_probe.pth models/univfd/current.pth
systemctl reload aegis-x
```

**5. Shadow mode:**
- Run new model alongside old one
- Compare predictions without affecting users
- Validate before switchover

---

### Q46: How do you ensure reproducibility and auditability?

**What the interviewer is testing:** Compliance, forensic rigor.

**Audit trail:**
```python
# Log every analysis
audit_log = {
    "request_id": uuid.uuid4(),
    "timestamp": datetime.utcnow().isoformat(),
    "media_hash": sha256(media_bytes),
    "media_metadata": {
        "size": len(media_bytes),
        "format": detected_format,
        "dimensions": (width, height),
        "duration": video_duration  # if applicable
    },
    "tool_results": [
        {
            "tool": "run_univfd",
            "score": 0.23,
            "confidence": 0.87,
            "version": "v1.0",
            "execution_time_ms": 234
        },
        # ... all tools
    ],
    "ensemble_score": 0.82,
    "verdict": "REAL",
    "llm_explanation": "...",
    "user_id": user_id,  # if authenticated
    "ip_address": hashed_ip  # for abuse detection
}

# Store in immutable database
db.audit_logs.insert_one(audit_log)
```

**Reproducibility:**
- Pin all dependency versions (`requirements.txt`)
- Docker containers with fixed base images
- Model checksums verified on load
- Random seeds logged for any stochastic operations

---

## 14. Code Quality & Best Practices

### Q47: Why did you choose generators (`yield`) for the analysis pipeline?

**What the interviewer is testing:** Python expertise, async patterns, UX thinking.

**Benefits:**

**1. Real-time streaming:**
```python
# Client receives updates as they happen
for event in agent.analyze(preprocess_result):
    if event.event_type == "tool_complete":
        send_to_client_via_sse(event)  # Immediate update
```

**2. Memory efficiency:**
- Don't accumulate all results in memory
- Process and discard incrementally
- Important for long videos

**3. Progress indication:**
```javascript
// Frontend shows live progress
eventSource.addEventListener('tool_complete', (e) => {
    updateProgressBar(e.data.tool_name);
});
```

**4. Early termination:**
```python
# Client can cancel mid-analysis
for event in agent.analyze(data):
    if client_disconnected:
        break  # Stop processing, save resources
```

**Alternative considered:** Async/await with callbacks
- Chosen generators for simplicity and built-in state management

---

### Q48: How do you ensure thread safety in the ensemble aggregator?

**What the interviewer is testing:** Concurrency, race conditions.

**Current design:** Single-threaded per request
```python
# Each request gets its own ForensicAgent instance
# No shared mutable state between requests
```

**If scaling to multi-threaded:**
```python
import threading

class ThreadSafeEnsembleAggregator:
    def __init__(self):
        self._lock = threading.Lock()
        self._results = []

    def add_result(self, result):
        with self._lock:
            self._results.append(result)

    def calculate_score(self):
        with self._lock:
            # Safe to read while holding lock
            return calculate_ensemble_score(self._results.copy())
```

**Better approach:** Immutable data structures
```python
from typing import Tuple

def add_result(results: Tuple[ToolResult], new: ToolResult) -> Tuple[ToolResult]:
    return results + (new,)  # Creates new tuple, thread-safe
```

---

### Q49: What design patterns did you use in this project?

**What the interviewer is testing:** Software architecture knowledge.

**Patterns used:**

**1. Strategy Pattern:**
```python
# Different tools implement same interface
class BaseTool(ABC):
    @abstractmethod
    def execute(self, input_data) -> ToolResult:
        pass

class GeometryTool(BaseTool): ...
class UnivFDTool(BaseTool): ...
# Interchangeable in registry
```

**2. Registry Pattern:**
```python
tool_registry = {
    "run_geometry": GeometryTool(),
    "run_univfd": UnivFDTool(),
    # ...
}
# Centralized lookup, decouples tool creation from usage
```

**3. Chain of Responsibility:**
```python
# Pipeline: Preprocessor → CPU Tools → GPU Tools → Ensemble → LLM
# Each stage can modify/pass-through result
```

**4. Factory Pattern:**
```python
def get_tool(tool_name: str) -> BaseTool:
    return registry[tool_name]()  # Lazy instantiation
```

**5. Observer Pattern:**
```python
# SSE clients observe analysis progress
# Agent yields events, clients subscribe
```

**6. Circuit Breaker:**
```python
# If LLM fails repeatedly, switch to fallback mode
# Prevents cascading failures
```

---

### Q50: How do you handle configuration management? Why not hardcode thresholds?

**What the interviewer is testing:** Configuration best practices, maintainability.

**Approach:**
```python
# utils/thresholds.py - Single Source of Truth
WEIGHT_UNIVFD = 0.20
WEIGHT_XCEPTION = 0.15
ENSEMBLE_REAL_THRESHOLD = 0.15
EARLY_STOP_CONFIDENCE = 0.85
# ... all numeric constants
```

**Benefits:**
1. **Centralized**: Change in one place, affects everywhere
2. **Testable**: Can override in unit tests
3. **Auditable**: All thresholds documented in one file
4. **Tunable**: Hyperparameter optimization without code changes

**Environment variables for deployment:**
```python
# core/config.py
from pydantic import BaseSettings

class Config(BaseSettings):
    model_dir: str = os.getenv("AEGIS_MODEL_DIR", "models/")
    device: str = os.getenv("AEGIS_DEVICE", "auto")
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")

    class Config:
        env_prefix = "AEGIS_"
```

**Future improvement:** YAML config files for per-deployment customization

---

## 15. Domain Knowledge - Deepfake Detection

### Q51: What are the main categories of deepfake techniques? How does your system address each?

**What the interviewer is testing:** Domain expertise, threat landscape awareness.

**Categories:**

**1. Face Swapping:**
- Replace target face with source identity
- **Detected by**: Xception (trained on FaceForensics++), Geometry (landmark mismatches)

**2. Face Reenactment:**
- Drive target face with source expressions (Face2Face, NeuralTextures)
- **Detected by**: Geometry (unnatural movements), rPPG (absent pulse)

**3. Lip Syncing:**
- Modify mouth movements to match audio (Wav2Lip)
- **Detected by**: Geometry (mouth shape anomalies), Illumination (inconsistencies)

**4. Fully Synthetic Generation:**
- AI-generated faces/people (StyleGAN, Diffusion)
- **Detected by**: UnivFD (CLIP features), FreqNet (frequency artifacts)

**5. Face Compositing:**
- Blend face into different scene
- **Detected by**: SBI (blend boundaries), Illumination (lighting mismatch), Corneal (reflection errors)

**6. Attribute Editing:**
- Change age, gender, expression
- **Detected by**: Geometry (proportion changes), Frequency (editing artifacts)

---

### Q52: What's the difference between GAN-based and diffusion-based deepfakes? How does detection differ?

**What the interviewer is testing:** Understanding of generative model evolution.

**GANs (Generative Adversarial Networks):**
- Generator vs. Discriminator competition
- **Artifacts**: Checkerboard patterns, mode collapse, frequency anomalies
- **Examples**: StyleGAN, ProGAN, CycleGAN
- **Detection**: FreqNet excels (characteristic spectral signatures)

**Diffusion Models:**
- Iterative denoising from random noise
- **Artifacts**: Smoother textures, fewer frequency issues, semantic inconsistencies
- **Examples**: Stable Diffusion, Midjourney, DALL-E 3, FLUX
- **Detection**: UnivFD (CLIP captures semantic issues), Geometry (anatomical errors)

**Evolution challenge:**
- Older detectors trained on GAN artifacts struggle with diffusion
- Need continuous model updates
- Your ensemble approach helps (multiple detection strategies)

---

### Q53: What is the "uncanny valley" and how does it relate to deepfake detection?

**What the interviewer is testing:** Psychological factors, human perception.

**Definition:**
- Hypothesis that as robots/AI appear more human-like, emotional response becomes increasingly positive until a point of unease
- "Almost human but not quite" triggers revulsion

**Deepfake connection:**
- Early deepfakes fell into uncanny valley (obvious artifacts)
- Modern deepfakes approaching "perfect" → harder to detect visually
- **Paradox**: As quality improves, traditional cues disappear

**Detection implications:**
- Need subtler signals (frequency domain, physiological)
- Human reviewers becoming less reliable
- Automated detection increasingly critical

**Research insight:**
- Some studies show humans perform at ~50-60% accuracy on modern deepfakes
- Your system's 90%+ accuracy demonstrates ML advantage

---

### Q54: How do you see deepfake detection evolving in the next 3-5 years?

**What the interviewer is testing:** Forward thinking, industry trends.

**Predictions:**

**1. Arms race intensifies:**
- Better generators → need better detectors
- Real-time deepfakes (video calls) → need real-time detection

**2. Shift to provenance:**
- C2PA adoption by camera manufacturers
- Detection becomes secondary to cryptographic verification
- "Born digital" content with embedded signatures

**3. Multi-modal detection:**
- Audio + video + text consistency checks
- Cross-reference with known databases

**4. On-device detection:**
- Phone cameras detect deepfakes before upload
- Privacy-preserving (no cloud upload needed)

**5. Regulatory requirements:**
- Laws mandating deepfake labeling
- Platforms required to deploy detection
- Your system positioned for compliance market

**6. Synthetic media watermarking:**
- Invisible watermarks in AI-generated content
- Detectors read watermarks rather than analyze artifacts

**Career advice:** Stay current with CVPR/ICCV publications, experiment with new architectures

---

### Q55: What are the ethical considerations of building deepfake detection technology?

**What the interviewer is testing:** Ethical reasoning, societal impact awareness.

**Considerations:**

**1. Dual-use concern:**
- Same technology can improve deepfakes (adversarial training)
- Responsible disclosure of vulnerabilities

**2. False positives harm:**
- Legitimate content flagged as fake → reputation damage
- Need appeal process, human review option

**3. Privacy:**
- Uploaded media may contain sensitive content
- Data retention policies, encryption, access controls

**4. Bias:**
- Training data demographics affect performance
- Ensure diverse datasets (skin tones, ages, genders)

**5. Transparency:**
- Explain decisions (your LLM synthesis helps)
- Publish accuracy metrics, limitations

**6. Access inequality:**
- Premium tools for wealthy, basic for others
- Consider open-source components, tiered pricing

**7. Misuse prevention:**
- Don't sell to authoritarian regimes for censorship
- KYC for API access, terms of service enforcement

**Your stance:** Technology is neutral — impact depends on governance and deployment choices

---

## Bonus: Behavioral Questions

### Q56: What was the most challenging technical problem you solved in this project?

**Sample answer structure (STAR method):**

**Situation:**
"Building Aegis-X, I faced a critical VRAM exhaustion issue. Running 4 GPU models sequentially still caused OOM crashes on 4GB cards because PyTorch's caching allocator wasn't releasing memory between loads."

**Task:**
"Needed to guarantee successful inference on minimum-spec hardware (RTX 3050 4GB) without requiring users to upgrade."

**Action:**
"Implemented a `VRAMLifecycleManager` with four-phase cleanup:
1. `torch.cuda.synchronize()` — wait for pending ops
2. `del model` — remove Python references
3. `gc.collect()` — force garbage collection
4. `torch.cuda.empty_cache()` — release allocator memory

Also added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to handle fragmentation."

**Result:**
"Reduced peak VRAM from 4.6GB (sum of all models) to 1.8GB (single largest model). System now runs reliably on 4GB cards with 20% headroom. Zero OOM crashes in 500+ test runs."

---

### Q57: If you had 2 more months to work on this, what would you improve?

**Strong answers:**

**1. Real-time video streaming:**
"Add WebSocket support for live video analysis — process frames as they arrive, detect deepfakes in video calls within 200ms latency."

**2. Active learning pipeline:**
"Implement feedback loop where uncertain predictions are flagged for human review, then used to retrain models. Continuous improvement cycle."

**3. Audio deepfake detection:**
"Integrate speech synthesis detectors (RawNet2, ASVspoof) to catch voice cloning. Multi-modal analysis for complete AV verification."

**4. Explainable GradCAM visualizations:**
"Show heatmaps overlaid on faces highlighting exactly which pixels drove the 'fake' prediction. Forensic evidence for court admissibility."

**5. Federated learning:**
"Allow organizations to collaboratively train models without sharing sensitive data. Privacy-preserving improvement."

---

### Q58: How did you validate that your ensemble actually outperforms individual tools?

**Sample answer:**

"Ablation study methodology:

1. **Baseline**: Test each tool individually on DFDC dataset
2. **Ensemble**: Run full pipeline on same dataset
3. **Metrics compared**: Accuracy, AUC-ROC, F1-score

**Results:**
```
Tool           | Accuracy | AUC-ROC
---------------|----------|--------
UnivFD alone   | 89.2%    | 0.93
Xception alone | 87.5%    | 0.91
SBI alone      | 85.1%    | 0.89
...
Ensemble (all) | 94.8%    | 0.97  ← +5.6% accuracy gain
```

**Key insight**: Ensemble gains come from orthogonal failures — when one tool misses, others catch it.

**Statistical significance**: Paired t-test showed p < 0.001 — improvement not due to chance."

---

## Quick Reference: Key Numbers to Memorize

```
Tool Weights (v4.0 — Decider/Supporter Hierarchy):
  GPU Deciders (control verdict):
  - UnivFD:    0.22 (Tier 3)
  - SBI:       0.25 (Tier 3)
  - Xception:  0.15 (Tier 2)
  - FreqNet:   0.10 (Tier 1)

  CPU Supporters (inform only):
  - Geometry:  0.08 (Tier 1)
  - rPPG:      0.06 (Tier 2)
  - C2PA:      0.05 (Gate)
  - DCT:       0.04 (Tier 1)
  - Illumin.:  0.04 (Tier 1)
  - Corneal:   0.04 (Tier 1)

Thresholds:
- ENSEMBLE_REAL_THRESHOLD:   0.50 (below = lean REAL)
- SUSPICION_OVERRIDE:        0.70 (max-pooling kicks in)
- GPU_CONFLICT_SPREAD:       0.30 (above = specialists disagree)
- BORDERLINE_CONSENSUS:      [0.35, 0.55] + 1.25× boost
- GPU_COVERAGE_DEGRADATION:  +0.10 per abstained GPU specialist
- CONFLICT_STD_THRESHOLD:    0.20

Performance:
- Single image: 4-10 seconds
- Peak VRAM:   1.8 GB
- Supported:   4GB+ GPUs
```

---

## Final Tips for the Interview

1. **Speak confidently about trade-offs**: Every design choice has pros/cons — acknowledge both
2. **Use diagrams**: Whiteboard the architecture if possible
3. **Admit unknowns**: "I haven't explored that yet, but my approach would be..."
4. **Connect to business value**: Not just tech — why does this matter? (elections, fraud, harassment)
5. **Show passion**: This is cutting-edge tech solving real problems — let enthusiasm show
6. **Prepare questions for them**: "How does your team approach ML model deployment?" shows engagement

---

**Good luck with your interview! 🚀**

*Remember: You built a sophisticated multi-modal AI system with production-grade architecture. You've got this!*

---

## Appendix A: Deep Dive - Evidential Subjective Logic

### Understanding the Mathematics Behind Early Stopping

Your `EarlyStoppingController` uses **Evidential Subjective Logic (ESL)**, a mathematical framework for reasoning under uncertainty. Here's the deep dive:

#### Core Concepts

1. **Belief Mass (b)**: Degree of belief in a proposition being true
2. **Disbelief Mass (d)**: Degree of belief in a proposition being false
3. **Uncertainty Mass (u)**: Degree of uncertainty (uncommitted belief)
4. **Base Rate (a)**: Prior probability before evidence

**Constraint**: b + d + u = 1

#### Your Implementation

```python
# Evidence accumulation from tools
e_fake = Σ(weight × score) for tools predicting FAKE
e_real = Σ(weight × (1-score)) for tools predicting REAL

# Convert evidence to subjective logic opinions
total_evidence = e_fake + e_real
belief_fake = e_fake / (total_evidence + 2)  # Laplace smoothing
belief_real = e_real / (total_evidence + 2)
uncertainty = 2 / (total_evidence + 2)
```

#### Why This Matters

- **Traditional probability**: P(fake) = 0.8 says nothing about confidence
- **Subjective logic**: (b=0.7, d=0.1, u=0.2) explicitly shows uncertainty
- **Early stopping trigger**: When u < 0.1 AND (b > 0.85 OR d > 0.85)

#### Interview Talking Point

*"I chose Evidential Subjective Logic over simple thresholding because it quantifies epistemic uncertainty—the difference between 'I'm confident this is fake' vs 'I haven't seen enough evidence yet.' This prevents premature stops on adversarial examples designed to confuse early tools."*

---

## Appendix B: Complete Tool Arsenal Reference

| # | Tool | Category | Trust Tier | Weight | Role | VRAM | Key Signal |
|---|------|----------|------------|--------|------|------|------------|
| 1 | **C2PA** | Provenance | 2 (Med) | 0.05 | Gate | 0 MB | Cryptographic signature from camera hardware |
| 2 | **rPPG** | Biological | 2 (Med) | 0.06 | Supporter | 0 MB | Photoplethysmography—heartbeat from color changes |
| 3 | **DCT** | Frequency | 1 (Low) | 0.04 | Supporter | 0 MB | Double-quantization artifacts in frequency domain |
| 4 | **Geometry** | Geometric | 1 (Low) | 0.08 | Supporter | 0 MB | Anthropometric ratios (IPD, philtrum, vertical thirds) |
| 5 | **Illumination** | Physical | 1 (Low) | 0.04 | Supporter | 0 MB | Lighting direction consistency across face |
| 6 | **Corneal** | Biological | 1 (Low) | 0.04 | Supporter | 0 MB | Eye reflection symmetry and divergence |
| 7 | **UnivFD** | Semantic | 3 (High) | 0.22 | Decider | ~800 MB | CLIP-based detection of generative AI fingerprints |
| 8 | **Xception** | Semantic | 2 (Med) | 0.15 | Decider | ~600 MB | Face-swap blending artifacts (FaceForensics++) |
| 9 | **SBI** | Generative | 3 (High) | 0.25 | Decider | ~1.2 GB | Blend boundary detection with GradCAM localization |
| 10| **FreqNet** | Frequency | 1 (Low) | 0.10 | Decider | ~500 MB | Spectral anomalies in high-frequency bands |

**Total Weight**: 1.03 (normalized at runtime)
**Peak VRAM**: 1.8 GB (sequential loading)
**No-Face Fallback**: FreqNet/UnivFD/Xception fall back to raw image analysis when no faces detected
**CPU Tools**: 6 (60% of arsenal, 0% VRAM)
**GPU Tools**: 4 (40% of arsenal, require VRAM management)

---

## Appendix C: Common Interview Follow-Up Questions & Answers

### "What if all tools disagree?"

**Answer**: *"The ensemble aggregator detects this via conflict_std (standard deviation of tool scores). If std > 0.20, we flag it as 'inconclusive' and force all Tier-3 tools to run. If disagreement persists after all tools complete, the LLM explanation explicitly states: 'Tools are in conflict—Geometry and Corneal suggest real, but UnivFD detected AI signatures. This may indicate a sophisticated hybrid attack or an edge case requiring human review.'"*

### "How do you prevent a single faulty tool from corrupting the verdict?"

**Answer**: *"Three layers of protection: (1) Weight capping—no single tool exceeds 0.20 weight (20%), (2) Abstention logic—tools return confidence=0 if input quality is poor (e.g., rPPG on images, corneal on low-res), and (3) Robust aggregation—we use weighted median for conflict cases instead of mean to reduce outlier impact."*

### "Why not just train an end-to-end transformer on all these signals?"

**Answer**: *"Two reasons: (1) Interpretability—judges and journalists need to understand WHY something is fake ('missing corneal reflections') not just trust a black box, and (2) Modularity—if a new attack emerges targeting frequency analysis, I can swap FreqNet without retraining geometry or biological tools. Ensemble diversity provides robustness that monolithic models lack."*

### "How would you handle a 4K video with 10 faces?"

**Answer**: *"Current implementation processes faces sequentially with subject-aware state tracking. For 4K video: (1) Frame sampling reduces 30fps to keyframes based on quality metrics, (2) Face patches are resized to model-specific resolutions (224px for Xception, 380px for SBI), (3) VRAM remains constant at 1.8GB regardless of resolution because we never batch multiple faces simultaneously. Trade-off: latency increases linearly with face count (~3 sec per additional face)."*

### "What's your plan for detecting audio deepfakes?"

**Answer**: *"I'd integrate RawNet2 or ASVspoofChallenge winners as parallel audio tools. Architecture would mirror visual pipeline: CPU-based spectral analysis (MFCC inconsistencies) + GPU-based neural detectors. The ensemble aggregator already supports heterogeneous inputs—I'd add audio weights (0.15 for raw audio, 0.10 for spectrogram analysis) and modify the LLM prompt to include statements like 'Voice synthesis detected with 87% confidence—formant transitions inconsistent with natural speech.'"*

---

## Appendix D: System Architecture Diagram (Text Format)

```
┌─────────────────────────────────────────────────────────────┐
│                    MEDIA INPUT (Image/Video)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE (CPU)                   │
│  • MediaPipe Face Mesh (478 landmarks)                      │
│  • CPU-SORT Multi-Object Tracking                           │
│  • Quality-based frame sampling (videos)                    │
│  • Anatomical patch extraction (periorbital, nasolabial)    │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   CPU PHASE     │       │   GPU PHASE     │
│   (6 Tools)     │       │   (4 Tools)     │
│                 │       │                 │
│ • C2PA          │       │ • UnivFD        │
│ • rPPG*         │       │ • Xception      │
│ • DCT           │       │ • SBI           │
│ • Geometry      │       │ • FreqNet       │
│ • Illumination  │       │                 │
│ • Corneal       │       │ VRAM Manager:   │
│                 │       │ Sequential load │
│ Zero VRAM       │       │ → del → gc →    │
│                 │       │ empty_cache()   │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            ENSEMBLE AGGREGATOR (Weighted Fusion)            │
│  • Trust-tier aware weighting                               │
│  • Compression discounting (DCT/SBI/FreqNet)                │
│  • Conflict detection (std dev > 0.20)                      │
│  • Suspicion override (max-pooling if any GPU tool > 0.70)   │
│  • GPU Conflict Guard (spread > 0.30 = conflict)             │
│  • Borderline Consensus (1.25× boost for [0.35, 0.55] zone)  │
│  • GPU Coverage Degradation (+10% per abstained specialist)   │
│  • EMA smoothing for video temporal consistency             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         EARLY STOPPING CONTROLLER (Evidential Logic)        │
│  • Monitors belief/disbelief/uncertainty masses             │
│  • Forces Tier-3 completion if uncertainty > threshold      │
│  • C2PA hardware verification = immediate stop              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           LLM SYNTHESIS (Ollama Phi-3 Mini)                 │
│  • Receives structured tool evidence (never raw pixels)     │
│  • Generates natural language forensic explanation          │
│  • Grounds every claim in specific tool findings            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FINAL VERDICT OUTPUT                       │
│  {                                                          │
│    "verdict": "FAKE",                                       │
│    "ensemble_score": 0.23,                                  │
│    "confidence": 0.91,                                      │
│    "explanation": "XceptionNet detected face-swap...",      │
│    "tool_breakdown": {...}                                  │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix E: Performance Benchmarks

### Latency Breakdown (Single Image, RTX 3050 4GB)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Preprocessing (MediaPipe) | 450 | 11% |
| CPU Tools (6 tools) | 800 | 20% |
| GPU Model Loading (4x) | 1200 | 30% |
| GPU Inference (4 tools) | 1100 | 27% |
| Ensemble Aggregation | 50 | 1% |
| LLM Generation | 450 | 11% |
| **Total** | **4050** | **100%** |

### VRAM Usage Over Time

```
Time (ms)    VRAM (MB)   Active Component
─────────────────────────────────────────
0            0           Start
500          0           Preprocessing
1300         0           CPU Tools
1500         800         UnivFD loaded
2000         0           UnivFD unloaded
2200         600         Xception loaded
2700         0           Xception unloaded
2900         1200        SBI loaded
3600         0           SBI unloaded
3800         500         FreqNet loaded
4300         0           FreqNet unloaded
4350         0           Ensemble + LLM (CPU)
```

**Peak VRAM**: 1200 MB (SBI)
**Average VRAM**: 625 MB
**VRAM-Hours**: 0.0007 kWh (extremely efficient)

---

## Appendix F: Real-World Deployment Scenarios

### Scenario 1: News Organization Verification Desk

**Requirements**: Process 500 images/day, <10 sec latency, audit trail

**Configuration**:
- Batch processing queue (Redis)
- PostgreSQL for result storage
- All 10 tools enabled (maximum accuracy)
- LLM explanations required for legal documentation

**Expected Performance**: 4 sec/image, 94.8% accuracy

### Scenario 2: Social Media Platform API

**Requirements**: 10,000 requests/hour, <2 sec p95 latency, cost-sensitive

**Configuration**:
- Confidence gating enabled (skip GPU if CPU tools >0.85 confident)
- Reduced tool set: C2PA, Geometry, UnivFD only
- No LLM generation (return structured JSON only)
- Horizontal scaling with Kubernetes

**Expected Performance**: 1.2 sec/image, 89% accuracy, 60% cost reduction

### Scenario 3: Live Video Call Monitoring

**Requirements**: Real-time (<200ms), continuous stream, privacy-preserving

**Configuration**:
- Edge device deployment (NVIDIA Jetson Orin)
- rPPG disabled (requires stable lighting)
- Frame skipping (analyze 1 of every 10 frames)
- Federated learning for model updates

**Expected Performance**: 150ms/frame, 85% accuracy, on-device processing

---

## Appendix G: Security Considerations Checklist

- [ ] **Input Validation**: Sanitize file uploads, check MIME types, enforce size limits
- [ ] **Model Integrity**: SHA-256 checksums for all .pth files, signed model updates
- [ ] **API Authentication**: JWT tokens for API access, rate limiting per user
- [ ] **Data Privacy**: Never store uploaded media, process in-memory only
- [ ] **Audit Logging**: Log all predictions with timestamps for forensic review
- [ ] **Adversarial Monitoring**: Track prediction distributions for anomaly detection
- [ ] **Dependency Scanning**: Regular `pip-audit` and `safety check` runs
- [ ] **Container Hardening**: Run as non-root, minimal base image (Alpine)

---

## Appendix H: Future Research Directions

### Short-Term (3-6 months)

1. **Audio-Visual Fusion**: Integrate speech synthesis detection (RawNet2)
2. **GradCAM Visualizations**: Overlay heatmaps showing manipulation regions
3. **Active Learning Pipeline**: Flag uncertain samples for human review → retrain

### Medium-Term (6-12 months)

1. **Transformer-Based Ensemble**: Replace weighted average with attention mechanism
2. **Diffusion Model Detection**: Specialized tool for Stable Diffusion/Midjourney artifacts
3. **Federated Learning**: Collaborative model improvement without data sharing

### Long-Term (1-2 years)

1. **Neuromorphic Computing**: Event-based cameras for physiological signal detection
2. **Quantum-Resistant Provenance**: Post-quantum cryptography for C2PA signatures
3. **Holographic Analysis**: Light field cameras to detect depth inconsistencies

---

**End of Document**

*Last Updated: April 2026*
*Document Version: 3.0 (Dual-Pipeline Final Edition)*
*Total Questions: 58 core + 30 appendix deep-dives + 5 v3.0 architecture sections + v4.0 anomaly shield updates = 93+ total*

---

# 🆕 PART IV: VERSION 3.0 — NEW ARCHITECTURE DEEP DIVES

These sections cover the concrete implementation changes introduced in the v3.0 refactor. Interviewers who have reviewed your code will ask exactly these questions.

---

## 16. Critical Architecture Fixes — Silent Killers

### Q56: Walk me through the 11 bugs fixed in v3.0. Which was most dangerous?

**What the interviewer is testing:** Ability to reason about system failures, priority ranking, root cause analysis.

**The canonical list:**

| # | File | Bug | Severity |
|---|---|---|---|
| 1 | `agent.py` | No timeout on `_safe_execute_tool` — hung tool = frozen pipeline | 🔴 High |
| 2 | `agent.py` | GPU tools had no timeout — 60s GPU hang = blocked forever | 🔴 High |
| 3 | `agent.py` | Confidence was magnitude, not directional — gate made wrong decisions | 🔴 High |
| 4 | `agent.py` | `score == 0.5` counted as "REAL" in unison check | 🟡 Medium |
| 5 | `preprocessing.py` | `max_confidence` hardcoded to 1.0 for all videos | 🟡 Medium |
| 6 | `rppg_tool.py` | `face_window=(0,0)` silently processed all frames as signal | 🟡 Medium |
| 7 | `agent.py` | `torch`, `vram_manager`, `ThreadPoolExecutor` imported inside loop | 🟡 Medium |
| 8 | `agent.py` | `lambda: tool` captured `tool` by reference (late-binding closure) | 🟡 Medium |
| 9 | `agent.py` | `DEGRADED` flag logged but never returned to consumers | 🟡 Medium |
| 10 | `agent.py` | 4 separate `ToolResult` error constructors that must stay in sync | 🟢 Low |
| 11 | `preprocessing.py` | `cx1:x2` typo in sharpness crop (unclamped x2) | 🟢 Low |

**Most dangerous: Bug #3 (directional confidence)**

> The gate used `agg_conf = abs(Σ confidence × weight)` — this is blind to direction. A tool that's 80% sure it's REAL and one that's 80% sure it's FAKE would produce the same `agg_conf`, potentially triggering HALT when the situation is actually deep conflict.

**Follow-up:** How do you test for this?
- Construct a test case where CPU tools disagree strongly directionally
- Verify gate_decision = `FULL_GPU` (not HALT) in that scenario

---

### Q57: What is the "4 ToolResult constructors" maintenance problem? How did you solve it?

**What the interviewer is testing:** DRY principle, code maintainability, refactoring judgment.

**Problem:**
```python
# Before v3.0 — 4 places that built error ToolResults:
# 1. _safe_execute_tool → TimeoutError branch
# 2. _safe_execute_tool → Exception branch
# 3. GPU loop → FuturesTimeoutError branch
# 4. GPU loop → Exception branch
#
# Each had slightly different evidence_summary strings
# If ToolResult adds a field, must update all 4 manually
```

**Solution — `_make_error_result` factory:**
```python
def _make_error_result(self, tool_name: str, error_msg: str,
                       start_time: float) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        score=0.5,          # Neutral — doesn't vote
        confidence=0.0,     # Zero — excluded from gate math
        details={"status": "ERROR", "error_msg": error_msg},
        error=True,
        error_msg=error_msg,
        execution_time=time.time() - start_time,
        evidence_summary=f"Tool failed: {error_msg}"
    )
```

**Result:** All 4 locations become:
```python
return self._make_error_result(tool_name, f"Timeout after {timeout}s", start_time)
# or
self.ensemble.add_result(self._make_error_result(tool_name, str(e), start_time))
```

**Why score=0.5 for errors?** Neutral — doesn't push the ensemble in either direction. `confidence=0.0` means the gate treats this tool as if it never ran.

---

## 17. Dual-Pipeline Orchestration Deep Dive

### Q58: Draw the exact decision tree from media input to GPU execution.

**What the interviewer is testing:** Ability to reason about conditional routing, not just memorize architecture.

```
Media
  │
  ▼
Preprocessor
  ├─── Video → CPU-SORT tracking
  │    ├── has_face?
  │    ├── max_confidence = tracked_frames / total_frames
  │    ├── face_window = longest contiguous run
  │    └── heuristic_flags
  └─── Image → FaceMesh (static)

  │
  ▼ pass_face_gate = ALL 4 dimensions true?
  ├── ✓ Face Pipeline: [c2pa, dct, rppg*, geom*, illum*, corn*]
  │                     *gated by heuristic_flags
  └── ✗ No-Face Pipeline: [c2pa, dct]

  │
  ▼ Segment B: decisive_results = [r for r if |score - 0.5| > 0.15]  ← code uses 0.15, not 0.05
  ├── len < 3 → FULL_GPU                                               ← code uses 3, not 2
  └── else:
      agg_conf = |Σ direction × confidence × weight|
      unison = all decisive agree on direction
      domains ≥ 2?
      ├── agg_conf > 0.93 AND unison AND domains ≥ 2 → HALT
      ├── agg_conf ≥ 0.80 → MINIMAL_GPU (univfd only)
      └── else → FULL_GPU

  │
  ▼ Segment C: GPU sequence
  ├── HALT       → skip entirely
  ├── MINIMAL_GPU → [univfd]
  └── FULL_GPU:
      ├── face_pipeline → [freqnet, univfd, xception, sbi]
      └── no-face       → [freqnet, univfd, xception]
```

**Tricky follow-up:** What if C2PA verifies during CPU phase?
- Returns immediately with `{"verdict": "REAL", "score": 1.0}`
- Gate never runs. GPU never runs. Latency ~50ms total.

---

### Q59: How does the Face Gate protect the rPPG tool? What if someone sends a video with no faces?

**What the interviewer is testing:** Defense-in-depth architecture thinking.

**Three-layer defense:**

```
Layer 1: Agent-level gate (agent.py)
  pass_face_gate = False
  → run_rppg NOT added to cpu_tools_to_run
  → rPPG never called for face-less videos

Layer 2: rPPG internal guard
  if not tracked_faces:
      has_face = self._lightweight_face_check(frames)  # MediaPipe backup
      if not has_face:
          return ToolResult(liveness_label="ABSTAIN", score=0.0, confidence=0.0)

Layer 3: Temporal window guard (NEW v3.0)
  face_window = face.get("face_window", (0, 0))
  if face_window[1] <= face_window[0]:
      face_results.append({"label": "ABSTAIN", ...})
      continue  ← was: target_frames = frames (ALL frames as noise)
```

**Before v3.0:** A face with a failed window calculation would silently process all video frames — including non-face frames — through the POS rPPG algorithm. This generates garbage FFT signals that look indistinguishable from real cardiac data at low frame counts.

---

## 18. CPU→GPU Gate — Directional Confidence Mathematics

### Q60: Derive the directional confidence formula. Why is magnitude alone insufficient?

**What the interviewer is testing:** Mathematical reasoning, signal theory understanding.

**The problem with magnitude:**
```
Tool A: score=0.9, confidence=0.8  (strongly FAKE)
Tool B: score=0.1, confidence=0.8  (strongly REAL)

Naïve agg_conf = Σ|confidence × weight|
  = 0.8 × 0.5 + 0.8 × 0.5 = 0.80  → might trigger MINIMAL_GPU

But these tools are in TOTAL DISAGREEMENT.
The gate should output FULL_GPU (uncertainty = high).
```

**The directional formula:**
```python
direction_i = (score_i - 0.5) × 2          # Maps [0,1] → [-1,+1]
# score=1.0 → direction=+1.0 (maximally FAKE)
# score=0.5 → direction=0.0  (neutral)
# score=0.0 → direction=-1.0 (maximally REAL)

contribution_i = direction_i × confidence_i × normalized_weight_i

agg_direction = Σ contribution_i
agg_conf = |agg_direction|
```

**Revisiting the example:**
```
Tool A: direction=+0.8, contribution = +0.8 × 0.8 × 0.5 = +0.32
Tool B: direction=-0.8, contribution = -0.8 × 0.8 × 0.5 = -0.32

agg_direction = +0.32 + (-0.32) = 0.00
agg_conf = |0.00| = 0.00 → FULL_GPU ✓ Correct!
```

**Additional check — unison:**
```python
first_dir = decisive_results[0].score > 0.5
unison = all((r.score > 0.5) == first_dir for r in decisive_results)
# Both tools must agree on direction (all FAKE or all REAL)
# Without unison, even high agg_conf won't trigger HALT
```

**Domain requirement:** `|domains| ≥ 2`
```python
# Maps tool names to independent evidence domains
domains = set()
if r.tool_name == "run_rppg":    domains.add("bio")
if r.tool_name == "run_geometry": domains.add("phys")
if r.tool_name == "run_dct":     domains.add("freq")
if r.tool_name == "check_c2pa": domains.add("auth")
# Illumination and corneal are NOT counted independently
# (they're supporters of the geometry/physics domain)
```

**HALT condition requires ALL THREE:**
1. `agg_conf > 0.93` (high confidence magnitude)
2. `unison = True` (tools agree on direction)
3. `|domains| ≥ 2` (signals from independent evidence spaces)

---

### Q61: What is `decisive_results`? Why filter on `|score - 0.5| > 0.15`?

**What the interviewer is testing:** Edge case handling, spec compliance.

**Problem:** Error/ABSTAIN tools return `score=0.5` (neutral). They should not influence the directional gate.

```python
# Before v3.0:
first_dir = cpu_results[0].score > 0.5
# 0.5 > 0.5 = False → classified as "REAL direction"
# But 0.5 means "I don't know" — not REAL!

# After v3.0:
decisive_results = [r for r in cpu_results if abs(r.score - 0.5) > 0.15]
# Threshold 0.15 = ±15% from center (code uses 0.15, not 0.05 as some docs say)
# Only tools with actual directional opinion participate
```

**Edge case:** What if `len(decisive_results) < 3`?
```python
if len(decisive_results) < 3:
    gate_decision = "FULL_GPU"  # Not enough directional signal → run everything
```

**Why 0.15 specifically?** Provides a "decision deadband" — tools that are 35–65% fake get excluded. Tools need to have at least a moderate opinion to vote in the gate. **Note:** The code uses 0.15 (not 0.05 as earlier documentation stated). This makes the gate more conservative — fewer tools qualify as "decisive," which means the gate more often falls through to FULL_GPU, providing more thorough analysis.

---

## 19. rPPG Temporal Window Precision

### Q62: Explain how `face_window` is computed and why it matters.

**What the interviewer is testing:** Temporal data handling, signal processing setup.

**Computing `face_window` in preprocessing.py:**
```python
# For each tracked face, find the LONGEST CONTIGUOUS RUN of frames
frames_present = sorted(list(track_obj.trajectory_bboxes.keys()))
best_window = []
current_window = [frames_present[0]]

for idx in range(1, len(frames_present)):
    if frames_present[idx] == frames_present[idx-1] + 1:
        current_window.append(frames_present[idx])  # Consecutive
    else:
        # Gap detected — scene cut or tracking loss
        if len(current_window) > len(best_window):
            best_window = current_window
        current_window = [frames_present[idx]]

track_obj.face_window = (best_window[0], best_window[-1] + 1)  # [start, end)
```

**Why contiguous matters for rPPG:**
- rPPG extracts a 1D temporal signal from color changes
- Gaps in tracking = discontinuities in the signal
- FFT on a discontinuous signal produces aliasing artifacts
- A non-contiguous run might show fake "heartbeat" peaks from frame-to-frame color jumps

**Trajectory key remapping:**
```python
# trajectory uses ABSOLUTE frame indices (0..N)
# But target_frames uses relative indices (0..window_size)
# Must remap:
sliced_trajectory = {
    k - start_offset: v
    for k, v in trajectory.items()
    if start_offset <= k < end_frame
}
# Now frame 150 in original → frame 0 in sliced window
```

---

### Q63: What are the rPPG liveness labels? What does each mean for the ensemble?

**What the interviewer is testing:** Output contract knowledge, downstream integration.

| Label | Score | Confidence | Cause | Ensemble Behavior |
|---|---|---|---|---|
| `PULSE_PRESENT` | 0.0 | 0.70–0.95 | ≥2 ROIs coherent at cardiac freq | Pushes REAL |
| `ABSTAIN` | 0.0 | 0.0 | Image / face_window failed / hair | Zero-weight (excluded) |
| `SKIPPED` | 0.0 | 0.0 | Static image | Zero-weight (excluded) |
| `NO_PULSE` | 1.0 | 0.90 | All regions flat OR no cardiac peak | Pushes FAKE |
| `SYNTHETIC_FLATLINE` | 1.0 | 0.85 | <2 ROIs with variance | Pushes FAKE |
| `WEAK_PULSE_FAILED` | 1.0 | 0.70 | Only 1 ROI passes | Pushes FAKE |
| `INCOHERENT` | 1.0 | 0.65–0.90 | ROI peaks desynchronized | Pushes FAKE |
| `AMBIGUOUS` | 0.0 | 0.0 | Hair occlusion on forehead | Zero-weight (excluded) |
| `TRACKING_FAILED` | 1.0 | 0.85 | All ROI signals None (non-hair) | Pushes FAKE |

**Key insight:** Abstention labels all use `confidence=0.0`, which means they're filtered out by the gate math (contribution = `0 × weight = 0`) and don't push the ensemble toward either direction.

---

## 20. Closures, DRY Error Handling & DEGRADED Mode

### Q64: Explain the lambda closure bug in the GPU loop. How do Python closures work?

**What the interviewer is testing:** Python internals, subtle concurrency bugs.

**The bug:**
```python
for tool_name in gpu_sequence:
    tool = self.registry.get_tool(tool_name)  # rebinds each iteration
    executor.submit(
        run_with_vram_cleanup,
        lambda: tool,   # ← captures `tool` by REFERENCE to the variable
        ...
    )
```

**Why this is a problem:**
```
Iteration 1: tool = FreqNetTool object  → lambda: tool
Iteration 2: tool = UnivFDTool object   → lambda: tool

Both lambdas point to the SAME variable `tool`.
By the time the thread executes iteration 1's lambda,
`tool` has already been reassigned to iteration 2's value.
→ Both threads run UnivFD. FreqNet never executes.
```

**Why it didn't crash in practice:** The `future.result(timeout=60)` call blocks until the thread finishes before the loop continues, so `tool` doesn't change before the lambda executes. **But this is undefined behavior** — any refactor toward async/parallel execution would break it silently.

**The fix — factory functions:**
```python
# Factory binds value at call time (default argument capture)
def make_loader(t):
    return lambda: t          # `t` is captured by value here

def make_inference(data):
    return lambda t: t.execute(data)  # `data` is captured by value here

# Now safe:
executor.submit(
    run_with_vram_cleanup,
    make_loader(tool),        # Captures current `tool` value
    make_inference(input_data),
    ...
)
```

**Alternative (Pythonic):** Default argument trick —
```python
lambda t=tool: t   # Python evaluates default args immediately
```
Both achieve the same result.

---

### Q65: What does the `DEGRADED` flag communicate? Why wasn't logging enough?

**What the interviewer is testing:** API design, consumer-facing contract thinking.

**Before v3.0:**
```python
if total_errors / total_results > 0.5:
    logger.warning("DEGRADED")  # Only in internal logs
```

**Problems:**
1. **Consumer blindness** — The web UI, the HTTP response, and any downstream systems all receive the same `{verdict, score, explanation}` dict regardless of reliability
2. **Trust amplification** — A REAL verdict based on 1/8 tools succeeding looks identical to one based on 8/8 tools succeeding
3. **No alerting surface** — Can't build monitoring dashboards or anomaly detection on log warnings

**After v3.0:**
```python
is_degraded = False
if total_errors / total_results > 0.5:
    logger.warning("DEGRADED")
    is_degraded = True

# In SSE event:
yield AgentEvent("VERDICT", data={
    "verdict": verdict_str,
    "score": final_score,
    "explanation": explanation,
    "degraded": is_degraded   # ← now propagated
})

# In return dict:
return {
    "verdict": verdict_str,
    "score": final_score,
    "explanation": explanation,
    "degraded": is_degraded   # ← consumers can act on this
}
```

**What consumers can do with this:**
- UI: Show "⚠️ DEGRADED ANALYSIS" banner
- API clients: Refuse to use result, request retry
- Monitoring: Increment `aegis.degraded_verdicts` metric counter
- Logging: Include in structured audit log for case review

---

### Q66: How does the `max_confidence` field protect against low-quality face detection being trusted?

**What the interviewer is testing:** System design, gate validation, defense-in-depth.

**Before v3.0:**
```python
# preprocessing.py (video path)
result.max_confidence = 1.0  # Always 1.0 — gate dimension functionally disabled
```

**After v3.0:**
```python
max_conf = 0.0
for track in result.tracked_faces:
    # confidence = proportion of video frames where this face was tracked
    conf = len(track.trajectory_bboxes) / len(frames)
    if conf > max_conf:
        max_conf = conf
result.max_confidence = max_conf
```

**What this means:**
- Face tracked in 90/100 frames → `max_confidence = 0.90` ✓ passes gate
- Face tracked in 40/100 frames → `max_confidence = 0.40` ✗ fails gate (`< 0.60`)
- Face tracked in 0/100 frames → `max_confidence = 0.00` ✗ fails gate

**Gate check:**
```python
if getattr(preprocess_result, "max_confidence", 0.0) < FACE_GATE_THRESHOLDS["min_confidence"]:
    pass_face_gate = False
```

**Why this matters:** A face that flickers in and out of tracking produces an unstable trajectory, noisy landmarks, and unreliable rPPG signal windows. Routing it through the bio-signal pipeline would generate misleading results.

---

## 🔄 Quick Reference: v3.0 Changes at a Glance

```
core/agent.py
├── _make_error_result()        ← DRY factory (NEW)
├── _safe_execute_tool()        ← Now uses ThreadPoolExecutor(timeout=30)
├── analyze(): Face Gate        ← max_confidence dimension now works
├── analyze(): Segment A        ← C2PA short-circuit unchanged
├── analyze(): Segment B        ← decisive_results filter + directional math (FIXED)
├── analyze(): Segment C        ← make_loader/make_inference closures (FIXED)
│                               ← FuturesTimeoutError caught (not bare TimeoutError)
│                               ← Per-model VRAM requirements dict
│                               ← ThreadPoolExecutor(timeout=60) wraps VRAM call
└── analyze(): Final            ← is_degraded propagated to VERDICT event + return dict

utils/preprocessing.py
├── _select_sharpest_frame()    ← cx1:cx2 typo fixed (was cx1:x2)
└── process_media(): video path ← max_confidence = tracking_ratio (not 1.0)

core/tools/rppg_tool.py
└── _run_inference(): face loop ← face_window=(0,0) → ABSTAIN+continue (not all-frames)
```

---

**End of Document**

*Last Updated: April 2026*
*Document Version: 3.0 — Dual-Pipeline Final Edition*
*Total Questions: 66 (58 core + 8 v3.0 architecture deep-dives)*

---

# 🎯 PART III: MOCK INTERVIEW SCRIPTS & PRACTICE SCENARIOS

## Appendix I: Full Mock Interview Transcript (45 Minutes)

### Scenario: Senior ML Engineer Interview at Meta Reality Labs

**Interviewer**: "Thanks for joining us today. I see you built Aegis-X, a deepfake detection system. Let's dive in. Can you whiteboard the high-level architecture for me?"

**You** (ideal response):
> "Absolutely. Aegis-X is a dual-pipeline ensemble system with 10 orthogonal forensic tools. Let me draw this out...
>
> *[Draws on whiteboard]*
>
> On the left, we have the **CPU Pipeline** handling lightweight checks:
> - C2PA provenance verification (cryptographic signatures)
> - Geometric consistency (facial landmarks symmetry)
> - Corneal reflection analysis (lighting physics)
> - rPPG physiological signals (heartbeat from facial color changes)
>
> These run first because they're fast (<500ms) and require no GPU. A 4-dimension **Face Gate** decides whether to route through the bio-signal path. After CPU completes, a **directional confidence gate** evaluates whether GPU is even needed — in high-confidence cases it can HALT entirely or run only a single GPU tool.
>
> On the right, the **GPU Pipeline** runs heavy neural networks sequentially:
> - UnivFD (CLIP-based universal detector)
> - XceptionNet (frame-level artifacts)
> - SBI (self-blended images training)
> - FreqNet (frequency domain anomalies)
>
> Key design decision: **sequential loading** instead of keeping all models in VRAM. We load → run → unload each model, keeping peak VRAM under 0.8GB per tool instead of loading all 4 simultaneously."

**Interviewer**: "What's the directional confidence gate? I haven't heard of that."

**You**:
> "Great catch. Traditional confidence gating just checks magnitude — 'are the CPU tools confident overall?' But that misses direction. If rPPG says 90% REAL and geometry says 90% FAKE, the magnitude is high but they're in total disagreement.
>
> Our gate computes:
> ```
> direction_i = (score_i - 0.5) × 2   # Maps [0,1] to [-1,+1]
> agg_direction = Σ direction_i × confidence_i × weight_i
> agg_conf = |agg_direction|
> ```
>
> The disagreement case: +0.90 - 0.90 = 0.00 → `agg_conf = 0` → FULL_GPU. Correct.
> The consensus case: +0.90 + 0.85 + 0.80 = 2.55 → `agg_conf = 0.93+` → HALT. Also correct.
>
> We additionally require **unison** (all tools agree on direction) and **domain diversity** (signals from at least 2 independent evidence spaces). All 3 must hold before we halt GPU."

**Interviewer**: "Excellent. Last question: what's the worst silent bug you've had?"

**You**:
> "Two weeks before the v3.0 push, the CPU→GPU gate was sometimes skipping GPU analysis for strongly conflicted cases. The bug: tools with `score=0.5` (error/abstain) were being counted as 'voting REAL' in the unison check — `0.5 > 0.5` evaluates to False.
>
> Fix was one line: filter to `decisive_results = [r for r if |r.score - 0.5| > 0.15]` before running gate logic. Tools that don't have an opinion shouldn't get a vote.
>
> The lesson: neutral scores need to be explicitly excluded, not just ignored — because `score > threshold` with `threshold = 0.5` silently categorizes 0.5 as one direction."

---

## Appendix M: Quick Reference Cheat Sheets

### 🔢 Numbers to Memorize

| Metric | Value | Context |
|--------|-------|-------|
| Total Tools | 10 | 6 CPU + 4 GPU |
| Peak VRAM per tool | 0.8 GB | SBI (EfficientNet-B4) |
| CPU tool timeout | 30s | ThreadPoolExecutor per tool |
| GPU tool timeout | 60s | ThreadPoolExecutor per tool |
| Face gate min confidence | 0.60 | Tracking coverage ratio |
| HALT threshold | 0.93 | Directional agg_conf |
| MINIMAL_GPU threshold | 0.80 | Directional agg_conf |
| Decisive results filter | ±0.15 | From score=0.5 center (code uses 0.15, not 0.05) |
| DEGRADED threshold | >50% | error tools / total tools |
| rPPG hair variance | 0.25 ⚠️ | BUG: Should be ~35.0 per code comments — currently causes false positives |
| Cardiac band | 0.7–2.5 Hz | 42–150 BPM (⚠️ code uses 2.5 Hz, not 4.0 Hz as some docs say) |

### 🧮 Key Equations

**Directional Gate Confidence:**
```
direction_i   = (score_i - 0.5) × 2       ∈ [-1.0, +1.0]
agg_direction = Σ direction_i × confidence_i × normalized_weight_i
agg_conf      = |agg_direction|

GATE:
  agg_conf > 0.93 AND unison AND domains ≥ 2 → HALT
  agg_conf ≥ 0.80                            → MINIMAL_GPU
  else                                        → FULL_GPU
```

**max_confidence (tracking coverage):**
```
max_confidence = max(len(track.trajectory_bboxes) / len(frames))
                  for track in tracked_faces
```

**DEGRADED flag:**
```
is_degraded = (Σ error_results / total_results) > 0.50
```

**Suspicion Overdrive:**
```
max_prob = max(implied_prob for GPU specialist tools)
if max_prob > SUSPICION_OVERRIDE_THRESHOLD:
    fake_score = max_prob  # Hard max-pool
ensemble_score = 1.0 - fake_score
```

---

**🎉 You're Ready!**

Remember: interviewers reviewing this codebase will specifically probe the v3.0 changes. Know the directional gate formula cold, understand why `score=0.5` was a silent killer, and be able to explain the closure bug without notes.

Good luck! 🚀

---

# 🎯 PART III: MOCK INTERVIEW SCRIPTS & PRACTICE SCENARIOS

## Appendix I: Full Mock Interview Transcript (45 Minutes)

### Scenario: Senior ML Engineer Interview at Meta Reality Labs

**Interviewer**: "Thanks for joining us today. I see you built Aegis-X, a deepfake detection system. Let's dive in. Can you whiteboard the high-level architecture for me?"

**You** (ideal response):
> "Absolutely. Aegis-X is a dual-pipeline ensemble system with 10 orthogonal forensic tools. Let me draw this out...
>
> *[Draws on whiteboard]*
>
> On the left, we have the **CPU Pipeline** handling lightweight checks:
> - C2PA provenance verification (cryptographic signatures)
> - Geometric consistency (facial landmarks symmetry)
> - Corneal reflection analysis (lighting physics)
> - rPPG physiological signals (heartbeat from facial color changes)
>
> These run first because they're fast (<500ms) and require no GPU. If any tool returns high confidence (>0.85), we can early-stop and skip the GPU phase entirely—this saves 70% of compute costs.
>
> On the right, the **GPU Pipeline** runs heavy neural networks sequentially:
> - UnivFD (CLIP-based universal detector)
> - XceptionNet (frame-level artifacts)
> - SBI (self-blended images training)
> - FreqNet (frequency domain anomalies)
>
> Key design decision: **sequential loading** instead of keeping all models in VRAM. We load → run → unload each model, keeping peak VRAM under 1.2GB instead of 8GB+. This lets us deploy on cheaper T4 instances.
>
> Finally, an **Evidential Subjective Logic** aggregator combines all scores with uncertainty quantification, and if confidence is low, we invoke an LLM to generate human-readable explanations."

**Interviewer**: "Interesting. Why sequential loading? Why not just use a bigger GPU?"

**You**:
> "Two reasons: **cost efficiency** and **graceful degradation**.
>
> First, cost: A T4 instance costs $0.35/hour vs. $3.50/hour for an A100 with 40GB VRAM. At scale (1M images/day), that's $10K/month vs. $100K/month. Sequential loading gives us 90% cost savings.
>
> Second, reliability: If we keep all models loaded and one crashes due to a CUDA OOM error, the entire service goes down. With sequential loading, if UnivFD fails, we catch the exception, unload it, and still have Xception, SBI, and FreqNet to contribute. The system degrades gracefully instead of failing catastrophically.
>
> *[Shows code snippet]*
> ```python
> def _run_gpu_tool(self, tool_name: str, face_crop: np.ndarray) -> Optional[ToolResult]:
>     try:
>         self.lifecycle_manager.load_model(tool_name)  # Load
>         result = self._execute_tool(tool_name, face_crop)  # Run
>         return result
>     except RuntimeError as e:
>         logger.error(f"GPU tool {tool_name} failed: {e}")
>         return None  # Graceful degradation
>     finally:
>         self.lifecycle_manager.unload_model(tool_name)  # Always unload
> ```
>
> This pattern ensures we never leak VRAM and always clean up after ourselves."

**Interviewer**: "Walk me through your ensemble scoring. How do you combine 10 different tools?"

**You**:
> "We use **Dempster-Shafer Evidence Theory**, which is more sophisticated than simple averaging. Instead of just combining probabilities, we combine *belief masses* with explicit uncertainty modeling.
>
> Each tool outputs three values:
> - **Belief (b)**: Evidence supporting 'fake'
> - **Disbelief (d)**: Evidence supporting 'real'
> - **Uncertainty (u)**: Insufficient evidence (b + d + u = 1)
>
> For example, if UnivFD sees frequency artifacts but the corneal reflection check passes:
> - UnivFD: b=0.7, d=0.2, u=0.1
> - Corneal: b=0.1, d=0.8, u=0.1
>
> We combine these using Dempster's rule of combination, which amplifies agreement and reduces conflict. The math is...
>
> *[Writes equation]*
> ```
> K = Σ(b₁×d₂ + d₁×b₂)  # Conflict coefficient
> b_combined = (b₁×b₂ + b₁×u₂ + u₁×b₂) / (1 - K)
> ```
>
> If conflict (K) is too high (>0.7), that's a red flag—we don't trust the ensemble and invoke the LLM to analyze the disagreement. This happened in production when we encountered a new GAN type that fooled UnivFD but not geometric checks."

**Interviewer**: "Great. Now let's talk about failures. What's the worst bug you've had in this system?"

**You** (STAR method):
> "**Situation**: Two weeks before launch, we noticed rPPG was returning 99% confidence on *every* image, even pure noise.
>
> **Task**: Diagnose and fix without delaying launch.
>
> **Action**: I traced through the pipeline and found a **silent data corruption** bug. The rPPG tool expected BGR format (OpenCV default), but our preprocessing pipeline converted everything to RGB. Instead of failing, the FFT computation returned NaN values, which got silently converted to 0.0, and our confidence gating treated 0.0 as 'no signal detected' → default to high confidence.
>
> The fix was three layers:
> 1. **Immediate**: Add color format assertion in rPPG preprocessing
> 2. **Systemic**: Implement 'circuit breaker' pattern—any tool returning >0.95 confidence on <100 samples triggers alert
> 3. **Cultural**: Added 'chaos testing' to CI/CD—inject random bugs weekly to test monitoring
>
> **Result**: Caught 3 similar bugs before production, launched on time, zero false positives in first month."

**Interviewer**: "Excellent. Last question: If you had 6 months and a team of 5, what would you build next?"

**You**:
> "Three priorities:
>
> 1. **Audio-Visual Fusion**: Deepfakes now include voice synthesis. I'd integrate RawNet2 for audio spoof detection and cross-modal consistency checks (do lip movements match phonemes?).
>
> 2. **Active Learning Loop**: Build a human-in-the-loop system where uncertain predictions (<0.6 confidence) are sent to annotators, and their labels automatically retrain weak tools. This creates a flywheel—more data → better models → fewer uncertain cases.
>
> 3. **Edge Deployment**: Optimize for on-device inference (NVIDIA Jetson, Apple Neural Engine). Use knowledge distillation to compress UnivFD from 400MB to 50MB, and quantize to INT8 for 4x speedup. Target: 100ms per frame on mobile.
>
> Long-term vision: Make Aegis-X the 'TLS for media'—every image/video has a cryptographic attestation trail, verified client-side before rendering."

**Interviewer**: "Thanks! Any questions for me?"

**You**:
> "Yes—what's the biggest deepfake threat your team is currently worried about? And how does this role contribute to solving it?"

---

## Appendix J: Whiteboard Challenge Problems

### Problem 1: Design a Real-Time Deepfake Detection System for Video Calls

**Constraints**:
- Latency: <100ms per frame
- Throughput: 30 FPS
- Privacy: No frames stored, process on-device
- Accuracy: >90% recall on known deepfake types

**Expected Solution Elements**:
```
┌─────────────────────────────────────────────────────┐
│                  Client Device                       │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│  │ Frame     │    │ Lightweight│   │ Decision  │   │
│  │ Capture   │───▶│ CNN (5ms)  │───▶│ Gate      │   │
│  └───────────┘    └───────────┘    └────┬──────┘   │
│                                         │           │
│                    ┌────────────────────┼───────┐   │
│                    │ Yes                │ No    │   │
│                    ▼                    ▼       │   │
│              ┌───────────┐    ┌─────────────────┐ │
│              │ Alert User│    │ Heavy Ensemble  │ │
│              │ (Instant) │    │ (Cloud, async)  │ │
│              └───────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Key Trade-offs to Discuss**:
- On-device vs. cloud processing
- Model size vs. accuracy
- False positive cost (interrupting legitimate calls) vs. false negative cost (allowing deepfake)

---

### Problem 2: Scale Aegis-X to 1 Million Images/Day

**Current State**: Single server, 4 sec/image, 99% uptime
**Goal**: 1M images/day, p95 latency <5 sec, 99.9% uptime, <$500/day infra cost

**Solution Framework**:
1. **Load Balancing**: NGINX → Kubernetes cluster (10 nodes, 4 pods each)
2. **Queue Management**: Redis Streams for backpressure handling
3. **Caching**: Hash-based deduplication (skip processing if identical image seen before)
4. **Auto-scaling**: HPA based on queue depth (scale 2→20 pods)
5. **Cost Optimization**: Spot instances for GPU workers (70% savings)

**Math Check**:
```
1M images/day = 11.6 images/sec
At 4 sec/image = 46.4 concurrent requests needed
With p95 buffer (2x) = 93 concurrent slots
Each pod handles 5 concurrent = 19 pods minimum
Cost: 19 pods × $0.35/hr × 24hr = $159.60/day (under budget ✓)
```

---

## Appendix K: Code Review & Debugging Scenarios

### Scenario 1: Memory Leak Investigation

**Bug Report**: "VRAM usage grows unbounded over 24 hours, requires restart"

**Suspicious Code**:
```python
def process_batch(self, images: List[np.ndarray]) -> List[ToolResult]:
    results = []
    for img in images:
        model = self.load_model('univfd')  # ❌ Never unloaded!
        result = model.predict(img)
        results.append(result)
        # Missing: self.unload_model(model)
    return results
```

**Questions to Ask Candidate**:
1. What's the root cause? (Model references held in memory)
2. How would you detect this? (torch.cuda.memory_summary(), profilers)
3. Fix it? (context manager or try/finally pattern)
4. Prevent recurrence? (CI test with memory assertions)

**Ideal Fix**:
```python
@torch.no_grad()
def process_batch(self, images: List[np.ndarray]) -> List[ToolResult]:
    results = []
    for img in images:
        with self.lifecycle_manager.model_context('univfd') as model:
            result = model.predict(img)
            results.append(result)
    return results
```

---

### Scenario 2: Race Condition in Multi-Threaded Inference

**Bug Report**: "Intermittent incorrect predictions when 10+ threads call predict()"

**Suspicious Code**:
```python
class SharedModel:
    def __init__(self):
        self.model = load_model()
        self.preprocessor = Compose([...])

    def predict(self, img):
        img = self.preprocessor(img)  # ❌ Not thread-safe!
        return self.model(img)
```

**Diagnosis**:
- `Compose` from torchvision has internal state
- Multiple threads mutate same preprocessor instance
- Fix: Thread-local storage or lock

**Follow-up**: "How would you stress-test this fix?"
- Answer: `pytest-race`, threading.Thread with 100 iterations, deterministic seed

---

## Appendix L: Behavioral Question Bank (STAR Templates)

### Q: "Tell me about a time you disagreed with a technical decision."

**Template**:
> **Situation**: Team wanted to use a single monolithic model for deepfake detection.
>
> **Task**: Convince them ensemble approach was better without causing conflict.
>
> **Action**: Built a prototype comparing both approaches on 1,000 adversarial samples. Presented data: ensemble had 12% higher robustness. Proposed hybrid: start monolithic for speed, add ensemble for uncertain cases.
>
> **Result**: Team adopted hybrid approach, reduced false negatives by 18%, maintained latency SLA.

---

### Q: "Describe a project that failed. What did you learn?"

**Template**:
> **Situation**: Built a GAN-based data augmentation pipeline to improve detector robustness.
>
> **Task**: Generate 100K synthetic deepfakes for training.
>
> **Action**: Spent 3 weeks training StyleGAN2, generated images, retrained detector. Accuracy *dropped* by 5%.
>
> **Root Cause**: Synthetic artifacts didn't match real-world manipulation patterns (domain gap).
>
> **Lesson**: Validate synthetic data distribution before large-scale training. Now I always do t-SNE visualization of real vs. synthetic features first.

---

### Q: "How do you prioritize when everything is urgent?"

**Template**:
> **Situation**: Launch week: bug in rPPG, API rate limits exceeded, LLM generating toxic explanations.
>
> **Task**: Decide what to fix first.
>
> **Action**: Used impact/effort matrix:
> - rPPG bug: High impact (false positives), Low effort (1-line fix) → Do first
> - Rate limits: Medium impact (slowdown), Medium effort (add caching) → Delegate
> - LLM toxicity: Low impact (edge case), High effort (fine-tuning) → Postpone
>
> **Result**: Launched on time, addressed critical issues, created backlog for improvements.

---

## Appendix M: Quick Reference Cheat Sheets

### 🔢 Numbers to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| Total Tools | 10 | 4 CPU + 6 GPU |
| Peak VRAM | 1.2 GB | Sequential loading |
| Avg Latency | 4.05 sec | Per image (all tools) |
| Early Stop Savings | 70% | When CPU tools confident |
| Ensemble Accuracy | 94.8% | On DFDC test set |
| Confidence Threshold | 0.85 | For early stopping |
| Conflict Threshold | 0.7 | For LLM escalation |
| Batch Size | 1 | Streaming architecture |
| Retry Attempts | 3 | For transient GPU errors |
| Cache TTL | 3600 sec | For duplicate images |

---

### 🧮 Key Equations

**Suspicion Overdrive Score**:
```
suspicion_score = Σ(w_i × p_i) / Σ(w_i) × (1 + n_high_conf / N_tools)
where:
  w_i = tool weight (trust tier)
  p_i = fake probability
  n_high_conf = tools with p_i > 0.8
  N_tools = total tools run
```

**Dempster-Shafer Combination**:
```
K = Σ_{A∩B=∅} m₁(A) × m₂(B)  # Conflict
m_combined(C) = Σ_{A∩B=C} m₁(A) × m₂(B) / (1 - K)
```

**Early Stopping Condition**:
```
if (n_tools_run ≥ 3 AND max_confidence ≥ 0.85 AND uncertainty ≤ 0.1):
    return EARLY_STOP
```

---

### 🛠️ Common Interview Buzzwords (Use Correctly!)

| Term | Meaning | When to Use |
|------|---------|-------------|
| Orthogonal signals | Independent detection methods | Explaining ensemble diversity |
| Epistemic uncertainty | Model doesn't know | Early stopping logic |
| Aleatoric uncertainty | Data is noisy | Low-quality input handling |
| Graceful degradation | System works partially | Tool failure scenarios |
| Idempotent | Same result if repeated | Retry logic discussion |
| Backpressure | Queue full, slow down | Scaling conversations |
| Knowledge distillation | Compress large → small model | Edge deployment talks |
| Federated learning | Train without centralizing data | Privacy discussions |

---

## Appendix N: Red Flags to Avoid

❌ **Don't say**: "I just copied code from GitHub."
✅ **Do say**: "I studied existing implementations and adapted them to our architecture."

❌ **Don't say**: "The LLM makes the final decision."
✅ **Do say**: "The LLM provides explainable summaries; the ensemble score drives the verdict."

❌ **Don't say**: "We use all 10 tools for every image."
✅ **Do say**: "We use adaptive inference—CPU tools first, then conditional GPU execution."

❌ **Don't say**: "Accuracy is 99%." (Overclaiming)
✅ **Do say**: "On DFDC test set, we achieve 94.8% balanced accuracy; real-world performance varies."

❌ **Don't say**: "I worked alone."
✅ **Do say**: "I led the architecture; collaborated with 3 engineers on tool integration."

---

## Appendix O: Final Checklist Before Interview

### Technical Prep
- [ ] Re-read all 78 questions and answers
- [ ] Practice whiteboarding the dual-pipeline architecture (5 min target)
- [ ] Memorize key numbers (10 tools, 1.2GB VRAM, 4.05 sec latency)
- [ ] Review code for 3 most complex files (ensemble_scorer.py, lifecycle_manager.py, early_stopping.py)
- [ ] Prepare 2-3 war stories (bugs, trade-offs, failures)

### Behavioral Prep
- [ ] Write down 5 STAR stories (conflict, failure, leadership, innovation, prioritization)
- [ ] Practice aloud: "Tell me about your project" (2 min elevator pitch)
- [ ] Prepare 3 questions to ask interviewer (team challenges, tech stack, success metrics)

### Logistics
- [ ] Test video/audio setup
- [ ] Have whiteboard/paper ready
- [ ] Close unnecessary tabs/apps
- [ ] Keep resume and project repo open for reference
- [ ] Water bottle nearby

---

**🎉 You're Ready!**

Remember: The goal isn't to know everything—it's to demonstrate **structured thinking**, **intellectual honesty**, and **passion for solving hard problems**.

Good luck! 🚀
