"""Forensic Summary — Tool outputs → Structured Phi-3 prompt.

Converts ensemble results into a natural language prompt that grounds
every claim in specific tool evidence. The LLM never sees raw pixels.
"""

from typing import Dict
from core.data_types import ToolResult
from utils.thresholds import REAL_THRESHOLD, FAKE_THRESHOLD


def build_phi3_prompt(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Build structured prompt for Phi-3 Mini reasoning.
    
    All scores are converted to REAL PROBABILITY (0.0 = definitely fake, 1.0 = definitely real).
    Per Spec Section 5: LLM receives only structured text, never raw images.
    """
    prompt_parts = []
    
    # ── CRITICAL SCORE DIRECTION BLOCK ──
    prompt_parts.append("=== CRITICAL: SCORE INTERPRETATION ===")
    prompt_parts.append("ALL SCORES BELOW ARE 'AUTHENTICITY SCORES' (Real Probability):")
    prompt_parts.append("  100% = Definitely REAL/AUTHENTIC")
    prompt_parts.append("  50%  = Uncertain / Inconclusive")
    prompt_parts.append("  0%   = Definitely FAKE/MANIPULATED")
    prompt_parts.append("A LOW score means this tool DETECTED manipulation artifacts.")
    prompt_parts.append("A HIGH score means this tool found NO manipulation artifacts.")
    prompt_parts.append("")
    
    # ── HEADER ──
    prompt_parts.append("=== AEGIS-X FORENSIC ANALYSIS ===\n")
    prompt_parts.append(f"Overall Authenticity Score: {ensemble_score:.1%}")
    prompt_parts.append(f"Verdict: {verdict}\n")
    
    # ── TOOL EVIDENCE (Each tool is a specialist) ──
    prompt_parts.append("=== SPECIALIST TOOL EVIDENCE ===")
    prompt_parts.append("Each tool below is an INDEPENDENT SPECIALIST that tests for a SPECIFIC type of manipulation.")
    prompt_parts.append("If one specialist finds manipulation, other specialists finding nothing does NOT cancel it out.\n")
    
    # Helper to convert fake_score to real probability and interpret
    def _interpret(score: float, tool_desc: str, high_label: str, low_label: str) -> str:
        real_prob = 1.0 - score
        pct = f"{real_prob:.0%}"
        if real_prob >= 0.70:
            interpretation = f"CLEAR — {high_label}"
        elif real_prob >= 0.45:
            interpretation = f"INCONCLUSIVE — borderline result"
        else:
            interpretation = f"SUSPICIOUS — {low_label}"
        return f"{tool_desc}: Authenticity {pct} → {interpretation}"
    
    # C2PA
    if "check_c2pa" in tool_results:
        r = tool_results["check_c2pa"]
        if r.details.get("c2pa_verified"):
            prompt_parts.append(f"✅ C2PA: Cryptographically verified (signer: {r.details.get('signer', 'Unknown')})")
        else:
            prompt_parts.append("⚪ C2PA: No provenance data found (unsigned media)")
    
    # rPPG
    if "run_rppg" in tool_results:
        r = tool_results["run_rppg"]
        label = r.details.get("liveness_label", "UNKNOWN")
        prompt_parts.append(f"💓 rPPG (Pulse Detection Specialist): {label} — {r.evidence_summary}")
    
    # DCT
    if "run_dct" in tool_results:
        r = tool_results["run_dct"]
        prompt_parts.append(f"📐 DCT (Compression Analysis): peak_ratio={r.details.get('peak_ratio', 0):.3f}")
        if r.details.get("grid_artifacts"):
            prompt_parts.append("  → Double-quantization artifacts detected (compression or tampering)")
    
    # GEOMETRY
    if "run_geometry" in tool_results:
        r = tool_results["run_geometry"]
        violations = r.details.get("violations", [])
        real_prob = 1.0 - r.score
        prompt_parts.append(_interpret(r.score, "📏 Geometry (Facial Proportion Specialist)", 
                                       "all anatomical ratios normal",
                                       "impossible facial proportions detected"))
        if violations:
            prompt_parts.append(f"  → Failed checks: {', '.join(violations)}")
    
    # Illumination
    if "run_illumination" in tool_results:
        r = tool_results["run_illumination"]
        prompt_parts.append(_interpret(r.score, "💡 Illumination (Lighting Consistency Specialist)",
                                       "consistent lighting across face",
                                       "lighting inconsistencies detected"))
        if r.confidence == 0.0:
            prompt_parts.append("  → ABSTAINED: insufficient directional lighting data")
    
    # Corneal
    if "run_corneal" in tool_results:
        r = tool_results["run_corneal"]
        prompt_parts.append(_interpret(r.score, "👁️ Corneal (Eye Reflection Specialist)",
                                       "consistent corneal reflections",
                                       "mismatched eye reflections detected"))
        if r.confidence == 0.0:
            prompt_parts.append("  → ABSTAINED: insufficient reflection data")
    
    # UnivFD
    if "run_univfd" in tool_results:
        r = tool_results["run_univfd"]
        prompt_parts.append(_interpret(r.score, "🧠 UnivFD (AI-Generated Image Specialist)",
                                       "no GAN/diffusion fingerprints found",
                                       "AI-generation signatures detected"))

    # Xception
    if "run_xception" in tool_results:
        r = tool_results["run_xception"]
        prompt_parts.append(_interpret(r.score, "🔬 XceptionNet (Face-Swap Specialist)",
                                       "natural facial blending patterns",
                                       "face-swap blending artifacts detected"))
    
    # SBI
    if "run_sbi" in tool_results:
        r = tool_results["run_sbi"]
        if r.details.get("boundary_detected"):
            prompt_parts.append(_interpret(r.score, "🪡 SBI (Blend Boundary Specialist)",
                                           "no composite boundaries",
                                           f"blend boundary at {r.details.get('boundary_region', 'unknown')}"))
        else:
            prompt_parts.append(_interpret(r.score, "🪡 SBI (Blend Boundary Specialist)",
                                           "no blend boundaries detected",
                                           "blend boundaries detected"))
    
    # FreqNet
    if "run_freqnet" in tool_results:
        r = tool_results["run_freqnet"]
        prompt_parts.append(_interpret(r.score, "📡 FreqNet (Frequency Fingerprint Specialist)",
                                       "normal frequency spectrum",
                                       "anomalous frequency patterns detected"))
    
    # ── REASONING RULES ──
    prompt_parts.append("\n=== REASONING RULES ===")
    prompt_parts.append("1. ALL scores above are AUTHENTICITY scores: high% = likely real, low% = likely fake.")
    prompt_parts.append("2. Each tool is an INDEPENDENT specialist. If XceptionNet says 38% authenticity (face-swap detected), that is a STRONG fake signal even if FreqNet says 100% (no GAN artifacts) — they test different things!")
    prompt_parts.append("3. Ground every claim in specific tool results.")
    prompt_parts.append("4. Explain any conflicts between tools naturally.")
    prompt_parts.append("5. Use probabilistic language ('suggests', 'consistent with') — avoid absolute certainty.")
    prompt_parts.append("6. Keep explanation under 150 words in plain language.")
    
    # ── OUTPUT FORMAT ──
    prompt_parts.append("\n=== OUTPUT FORMAT ===")
    prompt_parts.append("Return a single paragraph of plain text containing your explanation. DO NOT use JSON, markdown, or any structured format. Just write the natural language explanation directly.")
    
    return "\n".join(prompt_parts)


async def generate_verdict(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Generate LLM verdict with streaming support.
    
    Returns:
        str: Natural language explanation from Phi-3
    """
    from core.llm import stream_completion
    
    prompt = build_phi3_prompt(ensemble_score, tool_results, verdict)
    
    explanation = ""
    async for token in stream_completion(prompt, temperature=0.1, max_tokens=512):
        explanation += token
        # Yield token for UI streaming (handled by agent.py)
    
    return explanation