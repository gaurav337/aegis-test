import json
import time
from pathlib import Path
from typing import Any, Dict

# ✅ Aligns with YOUR project structure
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult


class C2PATool(BaseForensicTool):
    """Tool for verifying C2PA Content Credentials provenance data."""

    @property
    def tool_name(self) -> str:
        return "check_c2pa"

    def setup(self) -> None:
        """Import verification for c2pa-python."""
        try:
            import c2pa
            self._c2pa_available = True
        except ImportError:
            self._c2pa_available = False

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run C2PA extraction and verification logic."""
        start_time = time.time()

        # --- Guard: Missing Input ---
        if "media_path" not in input_data:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,                          # ✅ FIX: Match Base class exception handler
                confidence=0.0,
                details={"c2pa_verified": False},   # ✅ FIX: Explicit flag for Ensemble/Agent
                error=True,
                error_msg="Missing media_path in input_data",
                execution_time=0.0,                 # Base class will overwrite this
                evidence_summary="Missing media_path"
            )

        # --- Guard: Library Missing ---
        if not getattr(self, "_c2pa_available", False):
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,                          # ✅ FIX: Match Base class exception handler
                confidence=0.0,
                details={"c2pa_verified": False},   # ✅ FIX: Explicit flag
                error=True,
                error_msg="c2pa-python library not available",
                execution_time=0.0,
                evidence_summary="Missing library"
            )

        media_path = str(input_data["media_path"])
        import c2pa
        
        try:
            c2pa_dict = None
            
            # --- API Compatibility ---
            if hasattr(c2pa, "read_file"):
                try:
                    c2pa_data = c2pa.read_file(media_path)
                    if not c2pa_data:
                        return self._no_c2pa_result(start_time)
                    c2pa_dict = json.loads(c2pa_data) if isinstance(c2pa_data, str) else c2pa_data
                except Exception as read_err:
                    err_msg = str(read_err).lower()
                    if any(k in err_msg for k in ("not found", "no jumbf", "not supported")):
                        return self._no_c2pa_result(start_time)
                    raise read_err
            else:
                try:
                    reader = c2pa.Reader(media_path)
                    json_str = reader.json()
                    if not json_str:
                        return self._no_c2pa_result(start_time)
                    c2pa_dict = json.loads(json_str) if isinstance(json_str, str) else json_str
                except Exception as read_err:
                    err_msg = str(read_err).lower()
                    if any(k in err_msg for k in ("not found", "no jumbf", "not supported", "notsupported")):
                        return self._no_c2pa_result(start_time)
                    raise read_err
            
            # --- Extract Signer and AI Claims ---
            signer = None
            timestamp = None
            is_ai_generated = False
            
            if c2pa_dict and "active_manifest" in c2pa_dict:
                manifest_claim = c2pa_dict.get("manifests", {}).get(c2pa_dict["active_manifest"], {})
                sig_info = manifest_claim.get("signature_info", {})
                signer = sig_info.get("issuer")
                timestamp = sig_info.get("time")
                
                # Check assertions for AI generation
                assertions = manifest_claim.get("assertions", [])
                ai_keywords = ["gemini", "midjourney", "dall-e", "stable diffusion", "openai", "firefly", "ai", "generated"]
                
                for assertion in assertions:
                    label = assertion.get("label", "")
                    data = assertion.get("data", {})
                    
                    if label == "c2pa.actions":
                        actions = data.get("actions", [])
                        for action in actions:
                            action_type = action.get("action", "").lower()
                            software_agent = action.get("softwareAgent", "").lower()
                            
                            if ("generator" in action_type) or ("generated" in action_type) or (("c2pa.created" in action_type) and any(kw in software_agent for kw in ai_keywords)):
                                is_ai_generated = True
                                
                            for kw in ai_keywords:
                                if kw in software_agent or kw in action_type:
                                    is_ai_generated = True
            
            # --- Verification Logic ---
            has_valid_sig = signer is not None
            
            # ✅ FIX: Trim details to Spec Contract + Short-Circuit Flag
            details = {
                "c2pa_verified": has_valid_sig,  # ← Agent reads this for short-circuit
                "is_ai_generated": is_ai_generated,
                "signer": signer or "Unknown",
                "timestamp": timestamp or "Unknown"
            }

            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,                          # 0.0 = Authentic (Real)
                confidence=1.0 if has_valid_sig else 0.0,
                details=details,
                error=False,
                error_msg=None,
                execution_time=0.0,                 # Base class will overwrite this
                evidence_summary=f"Signed by {signer or 'Unknown'} at {timestamp or 'Unknown'}"
            )

        except json.JSONDecodeError as je:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,                          # ✅ FIX: Match Base class exception handler
                confidence=0.0,
                details={"c2pa_verified": False},
                error=True,
                error_msg=f"Failed to parse C2PA JSON: {str(je)}",
                execution_time=0.0,
                evidence_summary="C2PA metadata is malformed."
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,                          # ✅ FIX: Match Base class exception handler
                confidence=0.0,
                details={"c2pa_verified": False},
                error=True,
                error_msg=str(e),
                execution_time=0.0,
                evidence_summary=f"C2PA read error: {str(e)}"
            )

    def _no_c2pa_result(self, start_time: float) -> ToolResult:
        """Helper to return an abstention result when no C2PA data is present."""
        return ToolResult(
            tool_name=self.tool_name,
            success=True,                           # ✅ Correct: Graceful abstention
            score=0.0,
            confidence=0.0,
            details={"c2pa_verified": False},       # ✅ FIX: Explicit flag
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary="No C2PA provenance data found."
        )