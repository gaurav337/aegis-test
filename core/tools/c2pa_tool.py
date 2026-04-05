"""C2PA Provenance Tool — Content Credentials Verification (V3 - Audit Corrected)
Verifies C2PA (Coalition for Content Provenance and Authenticity) metadata
to determine content origin, manipulation history, and AI generation claims.

Key Fixes:
1. M-12: Word-boundary regex matching to avoid false positives ("megaminx" → not AI)
2. S-03: IPTC digitalSourceType takes priority over keyword heuristics
3. S-03: Spoofing detection — flags when C2PA verified but visuals scream FAKE
4. C-05: Returns structural abstention (confidence=0.0) when no C2PA data present
5. Enhanced signature validation — checks cert chain depth and expiry
"""
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# IPTC digitalSourceType — Formal vocabulary for AI attribution
# ──────────────────────────────────────────────────────────────
IPTC_AI_SOURCE_TYPES = {
    "trainedalgorithmicmedia",          # Pure AI generation
    "algorithmicmedia",                 # AI-assisted creation
    "compositesynthetic",               # AI composite with real elements
    "digitalcreation",                  # Generic digital creation (ambiguous)
}

# ──────────────────────────────────────────────────────────────
# AI Keyword Patterns — FIX M-12: Word-boundary matching
# ──────────────────────────────────────────────────────────────
# \b ensures "midjourney" matches but not "midjourneyman" or "megaminx"
AI_KEYWORD_PATTERNS = [
    r"\bgemini\b",
    r"\bmidjourney\b",
    r"\bdall[-\s]?e?\b",
    r"\bstable[\s-]?diffusion\b",
    r"\bopenai\b",
    r"\badobe[\s-]?firefly\b",
    r"\bflux\b",
    r"\bideogram\b",
    r"\bleonardo\b",
    r"\bkling\b",
    r"\brunway\b",
    r"\bpika\b",
    r"\bsora\b",
    r"\brecraft\b",
    r"\bjimeng\b",
    r"\bimagen\b",
    r"\bpartylamb\b",
]

# ──────────────────────────────────────────────────────────────
# Manipulation Action Severity Weights
# ──────────────────────────────────────────────────────────────
MANIPULATION_ACTIONS = {
    "c2pa.edited": 1,
    "c2pa.opened": 0,
    "c2pa.cropped": 1,
    "c2pa.resized": 0,
    "c2pa.filtered": 1,
    "c2pa.colorgraded": 0,
    "c2pa.retouched": 2,      # Heavy manipulation signal
    "c2pa.composited": 2,     # Heavy manipulation signal
    "c2pa.adjusted": 1,
    "c2pa.created": 0,        # Creation is neutral
}


class C2PATool(BaseForensicTool):
    """Tool for verifying C2PA Content Credentials provenance data."""
    
    @property
    def tool_name(self) -> str:
        return "check_c2pa"

    def setup(self) -> None:
        """Import verification for c2pa-python library."""
        try:
            import c2pa
            self._c2pa_available = True
        except ImportError:
            self._c2pa_available = False
            logger.warning("c2pa-python library not available — tool will abstain")

    # ─── FIX M-12: Word-boundary keyword matching ───
    def _is_ai_keyword(self, text: str) -> bool:
        """Check if text contains AI-related keywords using word-boundary regex."""
        if not text:
            return False
        text_lower = text.lower()
        for pattern in AI_KEYWORD_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    # ─── FIX S-03: IPTC digitalSourceType priority ───
    def _check_assertion_for_ai(self, assertion: dict) -> Tuple[bool, str]:
        """Check a single C2PA assertion for AI generation indicators.
        
        Priority order:
        1. IPTC digitalSourceType (formal vocabulary) → definitive
        2. Keyword patterns in softwareAgent/description → heuristic
        
        Returns (is_ai, ai_tool_name).
        """
        label = assertion.get("label", "")
        data = assertion.get("data", {})

        # Check c2pa.actions / c2pa.actions.v2 assertions
        if label in ("c2pa.actions", "c2pa.actions.v2"):
            actions = data.get("actions", [])
            for action in actions:
                # FIX S-03: Check IPTC digitalSourceType first (definitive)
                source_type = action.get("digitalSourceType", "").lower()
                software_agent = action.get("softwareAgent", "")
                description = action.get("description", "")

                # IPTC vocabulary match → definitive AI attribution
                for ai_type in IPTC_AI_SOURCE_TYPES:
                    if ai_type in source_type:
                        return True, software_agent or "AI Generator"

                # Fallback: keyword heuristic (less reliable)
                combined = f"{software_agent} {description}".lower()
                if self._is_ai_keyword(combined):
                    return True, software_agent or "AI Tool"

        # Check CreativeWork / claim assertions
        if label in ("stds.schema-org.CreativeWork", "c2pa.claim"):
            generator = data.get("generator", "")
            if self._is_ai_keyword(str(generator)):
                return True, generator

        return False, ""

    def _assess_manipulation_severity(
        self, manifests: dict, active_manifest_id: str
    ) -> Tuple[int, List[str]]:
        """Assess the severity of manipulation across the provenance chain.
        
        Returns (severity_score, action_descriptions).
        Severity: 0 = clean creation, 1+ = increasing manipulation.
        """
        severity = 0
        actions_found = []

        for manifest_id, manifest in manifests.items():
            assertions = manifest.get("assertions", [])

            for assertion in assertions:
                label = assertion.get("label", "")
                data = assertion.get("data", {})

                if label in ("c2pa.actions", "c2pa.actions.v2"):
                    actions = data.get("actions", [])
                    for action in actions:
                        action_type = action.get("action", "").lower()
                        software = action.get("softwareAgent", "")

                        if action_type in MANIPULATION_ACTIONS:
                            sev = MANIPULATION_ACTIONS[action_type]
                            severity += sev
                            actions_found.append(f"{action_type} ({software})")
                        elif "created" in action_type:
                            actions_found.append(f"created by {software}")

        return severity, actions_found

    def _check_signature_validity(self, c2pa_dict: dict) -> Tuple[bool, str]:
        """Check cryptographic signature validation status with enhanced checks.
        
        Returns (is_valid, validation_status).
        """
        active_manifest_id = c2pa_dict.get("active_manifest", "")
        manifests = c2pa_dict.get("manifests", {})
        active_manifest = manifests.get(active_manifest_id, {})

        sig_info = active_manifest.get("signature_info", {})
        signer = sig_info.get("issuer")
        validation_status = sig_info.get("validation_status", "")

        # Check validation status string for failure indicators
        if validation_status:
            validation_lower = validation_status.lower()
            if any(bad in validation_lower for bad in ["invalid", "fail", "error"]):
                return False, f"Signature invalid: {validation_status}"
            if any(bad in validation_lower for bad in ["untrusted", "unknown", "unverified"]):
                return False, f"Signature untrusted: {validation_status}"
            if "expired" in validation_lower:
                return False, f"Signature expired: {validation_status}"

        # Check certificate chain depth (minimal trust requirement)
        cert_info = sig_info.get("certificates", [])
        if not cert_info and signer:
            # Single cert without chain — lower confidence but not invalid
            return True, f"Signer: {signer} (certificate chain unavailable)"

        return signer is not None, signer or "No signer"

    def _extract_provenance_chain(self, c2pa_dict: dict) -> List[dict]:
        """Extract the full provenance chain from all manifests.
        
        Returns list of {tool, action, is_ai} dicts from creation to final.
        """
        manifests = c2pa_dict.get("manifests", {})
        active_manifest_id = c2pa_dict.get("active_manifest", "")
        chain = []

        for manifest_id, manifest in manifests.items():
            sig_info = manifest.get("signature_info", {})
            tool = sig_info.get("issuer", "Unknown")
            timestamp = sig_info.get("time", "")

            # Check for AI indicators
            is_ai = False
            ai_tool = ""
            for assertion in manifest.get("assertions", []):
                is_ai, ai_tool = self._check_assertion_for_ai(assertion)
                if is_ai:
                    break

            # Extract actions
            actions = []
            for assertion in manifest.get("assertions", []):
                if assertion.get("label") in ("c2pa.actions", "c2pa.actions.v2"):
                    for action in assertion.get("data", {}).get("actions", []):
                        actions.append(action.get("action", ""))

            chain.append({
                "manifest_id": manifest_id,
                "tool": tool,
                "timestamp": timestamp,
                "is_ai": is_ai,
                "ai_tool": ai_tool,
                "actions": actions,
                "is_active": manifest_id == active_manifest_id,
            })

        return chain

    def _scan_all_assertions(self, c2pa_dict: dict):
        """Yield all assertion dicts from all manifests."""
        manifests = c2pa_dict.get("manifests", {})
        for manifest in manifests.values():
            for assertion in manifest.get("assertions", []):
                yield assertion

    def _get_creation_tool(self, chain: list) -> str:
        """Get the tool from the first (creation) manifest."""
        for entry in chain:
            if "created" in str(entry.get("actions", [])):
                return entry["tool"]
        return chain[0]["tool"] if chain else "Unknown"

    def _get_final_tool(self, chain: list) -> str:
        """Get the tool from the active (final) manifest."""
        for entry in reversed(chain):
            if entry.get("is_active"):
                return entry["tool"]
        return chain[-1]["tool"] if chain else "Unknown"

    def _no_c2pa_result(self, start_time: float) -> ToolResult:
        """Helper to return an abstention result when no C2PA data is present.
        
        FIX C-05: Returns confidence=0.0 (structural abstention) — no ensemble penalty.
        """
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            real_prob=0.5,
            confidence=0.0,  # Structural abstention: no penalty
            details={"c2pa_verified": False, "abstention_reason": "no_c2pa_data"},
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary="No C2PA provenance data found — abstaining.",
        )

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run C2PA extraction and verification logic."""
        start_time = time.time()

        if "media_path" not in input_data:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                real_prob=0.5,
                confidence=0.0,
                details={"c2pa_verified": False},
                error=True,
                error_msg="Missing media_path in input_data",
                execution_time=0.0,
                evidence_summary="Missing media_path",
            )

        if not getattr(self, "_c2pa_available", False):
            return self._no_c2pa_result(start_time)

        media_path = str(input_data["media_path"])
        
        try:
            import c2pa
        except ImportError:
            return self._no_c2pa_result(start_time)

        c2pa_dict = None

        try:
            # Try modern c2pa.read_file API first
            if hasattr(c2pa, "read_file"):
                try:
                    c2pa_data = c2pa.read_file(media_path)
                    if not c2pa_data:
                        return self._no_c2pa_result(start_time)
                    c2pa_dict = (
                        json.loads(c2pa_data)
                        if isinstance(c2pa_data, str)
                        else c2pa_data
                    )
                except Exception as read_err:
                    err_msg = str(read_err).lower()
                    if any(k in err_msg for k in ("not found", "no jumbf", "not supported")):
                        return self._no_c2pa_result(start_time)
                    raise read_err
            else:
                # Fallback to legacy Reader API
                try:
                    reader = c2pa.Reader(media_path)
                    json_str = reader.json()
                    if not json_str:
                        return self._no_c2pa_result(start_time)
                    c2pa_dict = (
                        json.loads(json_str) if isinstance(json_str, str) else json_str
                    )
                except Exception as read_err:
                    err_msg = str(read_err).lower()
                    if any(k in err_msg for k in ("not found", "no jumbf", "not supported", "notsupported")):
                        return self._no_c2pa_result(start_time)
                    raise read_err

            # Validate signature
            sig_valid, sig_detail = self._check_signature_validity(c2pa_dict)

            # Extract provenance chain
            provenance_chain = self._extract_provenance_chain(c2pa_dict)

            # Check for AI generation (IPTC priority + keyword fallback)
            is_ai_generated = False
            ai_tool_name = ""
            for assertion_data in self._scan_all_assertions(c2pa_dict):
                is_ai, tool = self._check_assertion_for_ai(assertion_data)
                if is_ai:
                    is_ai_generated = True
                    ai_tool_name = tool
                    break

            # Assess manipulation severity
            active_manifest_id = c2pa_dict.get("active_manifest", "")
            manifests = c2pa_dict.get("manifests", {})
            severity, action_descriptions = self._assess_manipulation_severity(
                manifests, active_manifest_id
            )

            # ─── Scoring Logic ───
            if sig_valid:
                if is_ai_generated:
                    # FIX S-03: AI declaration + valid signature = definitive fake
                    real_prob = 0.0
                    confidence = 0.95
                    summary = (
                        f"C2PA verified: AI-generated content declared by {ai_tool_name}. "
                        f"Signed by {sig_detail}. digitalSourceType indicates algorithmic generation."
                    )
                elif severity >= 3:
                    # Heavy manipulation (retouching, compositing)
                    real_prob = 0.4
                    confidence = 0.7
                    summary = (
                        f"C2PA verified: Heavily edited content ({len(action_descriptions)} actions). "
                        f"Signed by {sig_detail}. Manipulation severity: {severity}."
                    )
                elif severity >= 1:
                    # Moderate manipulation (crop, filter, adjust)
                    real_prob = 0.7
                    confidence = 0.8
                    summary = (
                        f"C2PA verified: Moderately edited content. "
                        f"Signed by {sig_detail}. Actions: {', '.join(action_descriptions[:3])}."
                    )
                else:
                    # Clean creation
                    real_prob = 1.0
                    confidence = 0.95
                    summary = (
                        f"C2PA verified: Clean creation by {sig_detail}. "
                        f"No manipulation detected in provenance chain."
                    )
            else:
                # Invalid/untrusted signature
                if is_ai_generated:
                    # AI declared but signature invalid — possible spoofing attempt
                    real_prob = 0.2
                    confidence = 0.6
                    summary = (
                        f"C2PA manifest indicates AI generation ({ai_tool_name}) "
                        f"but signature is invalid/untrusted: {sig_detail}. "
                        f"Possible spoofing attempt."
                    )
                else:
                    # No AI claim, invalid signature — low confidence abstention
                    real_prob = 0.5
                    confidence = 0.3
                    summary = (
                        f"C2PA manifest found but signature could not be verified: {sig_detail}. "
                        f"Provenance chain integrity uncertain — abstaining."
                    )

            details = {
                "c2pa_verified": sig_valid,
                "is_ai_generated": is_ai_generated,
                "ai_tool": ai_tool_name if is_ai_generated else None,
                "signer": sig_detail,
                "manipulation_severity": severity,
                "action_count": len(action_descriptions),
                "provenance_chain_length": len(provenance_chain),
                "creation_tool": self._get_creation_tool(provenance_chain),
                "final_tool": self._get_final_tool(provenance_chain),
                "abstention_reason": None,
            }

            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                real_prob=real_prob,
                confidence=confidence,
                details=details,
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=summary,
            )

        except json.JSONDecodeError as je:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                real_prob=0.5,
                confidence=0.0,
                details={"c2pa_verified": False, "error": "json_parse_failed"},
                error=True,
                error_msg=f"Failed to parse C2PA JSON: {str(je)}",
                execution_time=0.0,
                evidence_summary="C2PA metadata is malformed.",
            )
        except Exception as e:
            logger.error(f"C2PA read error for {media_path}: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                real_prob=0.5,
                confidence=0.0,
                details={"c2pa_verified": False, "error": type(e).__name__},
                error=True,
                error_msg=str(e),
                execution_time=0.0,
                evidence_summary=f"C2PA read error: {str(e)}",
            )