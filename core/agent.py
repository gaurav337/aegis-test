"""Agent Loop — Dynamic tool orchestration with early stopping.

Generator-based execution with real-time UI feedback via AgentEvent yields.
"""

from typing import Dict, List, Any, Generator
import traceback
import time

from core.tools.registry import get_registry
from core.early_stopping import EarlyStoppingController, StopReason
from utils.ensemble import EnsembleAggregator
from core.llm import generate_verdict
from utils.logger import setup_logger
from core.data_types import ToolResult
import torch
from utils.vram_manager import run_with_vram_cleanup
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = setup_logger(__name__)

GPU_VRAM_REQUIREMENTS = {
    "run_freqnet": 0.4,
    "run_univfd": 0.6,
    "run_xception": 0.5,
    "run_sbi": 0.8,
}

FACE_GATE_THRESHOLDS = {
    "min_confidence": 0.60,
    "min_face_area_ratio": 0.01,
    "min_frames_with_faces": 0.30,
    "min_face_pixel_area": 2500,
}

class AgentEvent:
    """Real-time progress event for UI streaming."""
    def __init__(self, event_type: str, tool_name: str = None, data: dict = None):
        self.event_type = event_type
        self.tool_name = tool_name
        self.data = data or {}

class ForensicAgent:
    """Orchestrates forensic analysis with dynamic tool selection."""
    
    def __init__(self, config):
        self.config = config
        self.registry = get_registry()
        self.ensemble = EnsembleAggregator()
        self.esc = EarlyStoppingController(
            tool_registry=self.registry,
            thresholds=(0.5, 0.5)  # Ensembles are handled independently here
        )
        
    def _make_error_result(self, tool_name: str, error_msg: str, start_time: float) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            real_prob=0.5,
            confidence=0.0,
            details={"status": "ERROR", "error_msg": error_msg},
            error=True,
            error_msg=error_msg,
            execution_time=time.time() - start_time,
            evidence_summary=f"Tool failed: {error_msg}"
        )
        
    def _safe_execute_tool(self, tool_name: str, input_data: dict, timeout: int = 30) -> ToolResult:
        start_time = time.time()
        try:
            tool = self.registry.get_tool(tool_name)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.execute, input_data)
                result = future.result(timeout=timeout)
            return result
        except FuturesTimeoutError:
            logger.error(f"Timeout executing {tool_name} after {timeout}s")
            return self._make_error_result(tool_name, f"Timeout after {timeout}s", start_time)
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}\n{traceback.format_exc()}")
            return self._make_error_result(tool_name, str(e), start_time)

    def analyze(self, preprocess_result: Any, media_path: str = None, generate_explanation: bool = True) -> Generator[AgentEvent, Any, dict]:
        """
        Main analysis loop — orchestrates CPU → GPU Gate → GPU Phase → Ensemble → LLM.
        """
        self.ensemble = EnsembleAggregator()
        flags = getattr(preprocess_result, "heuristic_flags", [])
        
        input_data = {
            "media_path": media_path,
            "tracked_faces": preprocess_result.tracked_faces,
            "frames_30fps": preprocess_result.frames_30fps,
            "first_frame": getattr(preprocess_result, "first_frame", None),
            "original_media_type": getattr(preprocess_result, "original_media_type", "image"),
            "heuristic_flags": flags,
        }
        self.ensemble.flags = flags
        
        # Determine Face Gate
        face_detected = getattr(preprocess_result, "has_face", False)
        pass_face_gate = face_detected
        
        # Example validation against thresholds if a face is detected
        if face_detected:
            if getattr(preprocess_result, "max_confidence", 0.0) < FACE_GATE_THRESHOLDS["min_confidence"]:
                pass_face_gate = False
            if getattr(preprocess_result, "max_face_area_ratio", 0.0) < FACE_GATE_THRESHOLDS["min_face_area_ratio"]:
                pass_face_gate = False
            if getattr(preprocess_result, "frames_with_faces_pct", 0.0) < FACE_GATE_THRESHOLDS["min_frames_with_faces"]:
                pass_face_gate = False

        yield AgentEvent("PIPELINE_SELECTED", data={"face_pipeline": pass_face_gate})

        # ── SEGMENT A: CPU PHASE ──
        cpu_tools_to_run = ["check_c2pa", "run_dct"]
        if pass_face_gate:
            if not getattr(preprocess_result, "insufficient_temporal_data", False) and input_data["original_media_type"] != "image":
                cpu_tools_to_run.append("run_rppg")
            
            run_geo, run_illum, run_corn = True, True, True
            if "MOTION_BLUR" in flags or "OCCLUSION" in flags:
                run_geo, run_illum, run_corn = False, False, False
            elif "FACE_TOO_SMALL" in flags:
                run_corn = False
            elif "LOW_LIGHT" in flags:
                run_illum, run_corn = False, False
                
            if run_geo: cpu_tools_to_run.append("run_geometry")
            if run_illum: cpu_tools_to_run.append("run_illumination")
            if run_corn: cpu_tools_to_run.append("run_corneal")

        for tool_name in cpu_tools_to_run:
            yield AgentEvent("TOOL_STARTED", tool_name)
            result = self._safe_execute_tool(tool_name, input_data, timeout=30)
            self.ensemble.add_result(result)
            yield AgentEvent("tool_complete", tool_name, data={
                "success": result.success,
                "real_prob": result.real_prob,
                "confidence": result.confidence,
                "evidence_summary": result.evidence_summary,
                "error_msg": result.error_msg,
            })
            
            if tool_name == "check_c2pa" and result.success and result.details.get("c2pa_verified", False):
                is_ai = result.details.get("is_ai_generated", False)
                signer = result.details.get("signer", "Unknown")
                
                if is_ai:
                    yield AgentEvent("early_stop", data={"reason": "C2PA_AI_GENERATED", "confidence": 1.0})
                    yield AgentEvent("verdict", data={
                        "verdict": "FAKE",
                        "real_prob": 0.0,
                        "explanation": f"C2PA Content Credentials detected. The cryptographically signed provenance data explicitly states this media was AI-generated (signed by: {signer}).",
                        "degraded": False
                    })
                else:
                    yield AgentEvent("early_stop", data={"reason": "C2PA_REAL_IMAGE", "confidence": 1.0})
                    yield AgentEvent("verdict", data={
                        "verdict": "REAL",
                        "real_prob": 1.0,
                        "explanation": f"Cryptographically signed via C2PA Content Credentials, confirming authentic hardware capture (signed by: {signer}).",
                        "degraded": False
                    })
                return

        # ── SEGMENT B: CPU->GPU GATE ──
        # Calculate CPU phase confidence
        cpu_results = [r for name, r in self.ensemble.tool_results.items() if name in cpu_tools_to_run and r.success and not r.error and r.details.get("liveness_label") not in ("ABSTAIN", "ERROR")]
        
        decisive_results = [r for r in cpu_results if abs(r.real_prob - 0.5) > 0.15]
        
        gate_decision = "FULL_GPU"
        unison_agreement = False
        agg_conf = 0.0
        
        if len(decisive_results) < 3:
            gate_decision = "FULL_GPU"
        else:
            baseline_weights = {
                "run_rppg": 0.35,
                "run_geometry": 0.25,
                "run_dct": 0.15,
                "run_illumination": 0.10,
                "run_corneal": 0.10,
                "check_c2pa": 0.05
            }
            active_weights = {r.tool_name: baseline_weights.get(r.tool_name, 0.0) for r in decisive_results}
            total_active_weight = sum(active_weights.values())
            
            if total_active_weight > 0:
                normalized_weights = {k: v / total_active_weight for k, v in active_weights.items()}
                directional_scores = []
                for r in decisive_results:
                    weight = normalized_weights[r.tool_name]
                    # Positive means more fake-like; negative means more real-like.
                    direction = (0.5 - r.real_prob) * 2
                    directional_scores.append(direction * r.confidence * weight)
                agg_direction = sum(directional_scores)
                agg_conf = abs(agg_direction)
            
            # Check unison agreement
            first_dir = decisive_results[0].real_prob < 0.5
            unison = all((r.real_prob < 0.5) == first_dir for r in decisive_results)
            
            # Independent domains check
            domains = set()
            for r in decisive_results:
                if r.tool_name == "run_rppg": domains.add("bio_rppg")
                elif r.tool_name == "run_geometry": domains.add("phys")
                elif r.tool_name == "run_dct": domains.add("freq")
                elif r.tool_name == "check_c2pa": domains.add("auth")
                elif r.tool_name == "run_corneal": domains.add("bio_corn")
                elif r.tool_name == "run_illumination": domains.add("illum")
                
            if unison and len(domains) >= 2:
                unison_agreement = True
                
            if agg_conf > 0.93 and unison_agreement:
                gate_decision = "HALT"
            elif agg_conf >= 0.80:
                gate_decision = "MINIMAL_GPU"
            else:
                gate_decision = "FULL_GPU"
                
        yield AgentEvent("GATE_DECISION", data={"decision": gate_decision, "confidence": agg_conf, "unison": unison_agreement})

        # ── SEGMENT C: GPU PHASE ──
        if gate_decision != "HALT":
            gpu_sequence = []
            if not pass_face_gate:
                gpu_sequence = ["run_freqnet", "run_univfd", "run_xception"]
            else:
                gpu_sequence = ["run_freqnet", "run_univfd", "run_xception", "run_sbi"]
                
            if gate_decision == "MINIMAL_GPU":
                gpu_sequence = ["run_univfd"]
                
            for tool_name in gpu_sequence:
                yield AgentEvent("TOOL_STARTED", tool_name)
                start_time = time.time()
                try:
                    tool = self.registry.get_tool(tool_name)
                    if tool is None:
                        raise RuntimeError(f"Tool {tool_name} not found in registry")

                    # SubprocessToolProxy manages its own subprocess + VRAM internally.
                    # Do NOT wrap in run_with_vram_cleanup — that treats the proxy as an
                    # nn.Module and causes the 0MB CUDA log + VRAMLifecycleManager lock contention.
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(tool.execute, input_data)
                        result = future.result(timeout=90)  # 90s: model load + inference
                    
                    self.ensemble.add_result(result)
                    yield AgentEvent("tool_complete", tool_name, data={
                        "success": result.success,
                        "real_prob": result.real_prob,
                        "confidence": result.confidence,
                        "evidence_summary": result.evidence_summary,
                        "error_msg": result.error_msg,
                    })
                except FuturesTimeoutError:
                    logger.error(f"Timeout executing {tool_name} after 90s")
                    self.ensemble.add_result(self._make_error_result(tool_name, "Timeout after 90s", start_time))
                    yield AgentEvent("tool_complete", tool_name, data={"success": False, "real_prob": 0.5, "confidence": 0.0, "evidence_summary": "Tool timed out.", "error_msg": "Timeout after 90s"})
                except Exception as e:
                    logger.error(f"Error executing {tool_name}: {e}\n{traceback.format_exc()}")
                    self.ensemble.add_result(self._make_error_result(tool_name, str(e), start_time))
                    yield AgentEvent("tool_complete", tool_name, data={"success": False, "real_prob": 0.5, "confidence": 0.0, "evidence_summary": "Tool failed.", "error_msg": str(e)})

        # Check DEGRADED status if >50% of mapped tools errored out
        is_degraded = False
        total_errors = sum(1 for r in self.ensemble.tool_results.values() if r.error)
        if len(self.ensemble.tool_results) > 0 and total_errors / len(self.ensemble.tool_results) > 0.5:
            logger.warning("Agent Output flagged as DEGRADED due to excessive tool failures.")
            is_degraded = True

        # ── ENSEMBLE SCORING ──
        final_score = self.ensemble.get_final_score()
        verdict_str = self.ensemble.get_verdict()
        
        yield AgentEvent("llm_start")
        
        if generate_explanation:
            explanation = yield from generate_verdict(
                ensemble_score=final_score,
                tool_results=self.ensemble.tool_results,
                verdict=verdict_str,
                config=self.config
            )
        else:
            explanation = "[Explanation skipped during batch evaluation]"
            
        yield AgentEvent("verdict", data={
            "verdict": verdict_str,
            "real_prob": final_score,
            "explanation": explanation,
            "degraded": is_degraded
        })
        
        return {
            "verdict": verdict_str,
            "real_prob": final_score,
            "explanation": explanation,
            "degraded": is_degraded,
        }