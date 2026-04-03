"""
EarlyStoppingController: 2026 SOTA Standard for Agentic Early Stopping.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from core.tools.registry import ToolRegistry

from core.tools.registry import ToolSpec

logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Reasons for early stop or continue decisions."""
    CONTINUE_AMBIGUOUS = "Score is ambiguous, continue analysis."
    CONTINUE_ADVERSARIAL_CONFLICT = "Evidential conflict detected (Adversarial risk), forcing continue."
    CONTINUE_SECURITY_REQUIRED = "High confidence, but awaiting high-trust verification."
    HALT_C2PA_HARDWARE_SIGNED = "Cryptographically verified camera-original media."
    HALT_LOCKED_FAKE = "Mathematically locked FAKE verdict."
    HALT_LOCKED_REAL = "Mathematically locked REAL verdict."


@dataclass
class StopDecision:
    """Decision object returned by EarlyStoppingController."""
    should_stop: bool
    reason: StopReason
    confidence: float
    details: Optional[str] = None
    evidence_metrics: Optional[Dict[str, float]] = None


class EarlyStoppingController:
    """
    2026 SOTA Early Stopping Controller using Evidential Subjective Logic.
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",  # Use string annotation to avoid circular import
        thresholds: Tuple[float, float],
        conflict_threshold: float = 0.35,
        min_evidence_magnitude: float = 0.10
    ):
        """
        Initialize the EarlyStoppingController.

        Parameters
        ----------
        tool_registry : ToolRegistry
            Source of truth for tool weights and trust tiers.
        thresholds : Tuple[float, float]
            (real_threshold, fake_threshold) tuple.
        conflict_threshold : float, optional
            Conflict ratio above which to block early stop (default 0.35).
        min_evidence_magnitude : float, optional
            Minimum total evidence required to trigger conflict check (default 0.10).
        """
        self.registry = tool_registry
        self.tool_specs = tool_registry.get_all_tool_specs()
        self.total_system_weight = tool_registry.get_total_system_weight()
        self.real_threshold, self.fake_threshold = thresholds
        self.conflict_threshold = conflict_threshold
        self.min_evidence_magnitude = min_evidence_magnitude

        # Pre-compute high trust tool names for O(1) lookup
        self.high_trust_tools: Set[str] = set(
            tool_registry.get_high_trust_tools()
        )

    def evaluate(
        self,
        tool_scores: Dict[str, float],
        completed_tools: List[str],
        c2pa_hardware_verified: bool = False
    ) -> StopDecision:
        """
        Evaluate whether to stop early or continue analysis.

        Parameters
        ----------
        tool_scores : Dict[str, float]
            Dictionary of tool_name -> score for successfully executed tools.
        completed_tools : List[str]
            List of tools that have already executed (successfully or failed).
        c2pa_hardware_verified : bool, optional
            True if C2PA hardware enclave attestation passed.

        Returns
        -------
        StopDecision
            Decision object with should_stop, reason, confidence, and evidence_metrics.
        """
        # 1. C2PA Hardware Lock
        if c2pa_hardware_verified:
            return StopDecision(
                True,
                StopReason.HALT_C2PA_HARDWARE_SIGNED,
                1.0,
                evidence_metrics={"c2pa_verified": True}
            )

        # 2. Empty scores check
        if not tool_scores:
            return StopDecision(False, StopReason.CONTINUE_AMBIGUOUS, 0.0)

        # 3. Validate Tool Names
        valid_tools = {}
        for t, s in tool_scores.items():
            if t in self.tool_specs:
                valid_tools[t] = s
            else:
                logger.warning(f"Contract Violation: Tool '{t}' not in registry.")

        if len(valid_tools) != len(tool_scores):
            logger.warning(
                f"Tool Score Contract Violation: "
                f"{len(tool_scores) - len(valid_tools)} unknown tools ignored."
            )

        # 4. Calculate State Metrics
        weights_run = sum(self.tool_specs[t].weight for t in valid_tools.keys())
        
        # Get viable pending tools from registry (excludes failed tools)
        viable_pending_tools = self.registry.get_viable_pending_tools(completed_tools)
        weights_pending = sum(
            self.tool_specs[t].weight for t in viable_pending_tools if t in self.tool_specs
        )

        if weights_run == 0:
            logger.error("Critical Error: Valid tools have 0 total weight.")
            return StopDecision(False, StopReason.CONTINUE_AMBIGUOUS, 0.0)

        # Weighted Mean
        weighted_sum = sum(
            score * self.tool_specs[t].weight for t, score in valid_tools.items()
        )
        current_score = weighted_sum / weights_run

        # 5. Evidential Subjective Logic
        evidence_metrics: Dict[str, float] = {}

        if len(valid_tools) > 1:
            e_fake = sum(
                self.tool_specs[t].weight * max(0, score - 0.5) * 2
                for t, score in valid_tools.items()
            )
            e_real = sum(
                self.tool_specs[t].weight * max(0, 0.5 - score) * 2
                for t, score in valid_tools.items()
            )

            total_evidence = e_fake + e_real
            evidence_metrics = {
                "e_fake": e_fake,
                "e_real": e_real,
                "total_evidence": total_evidence
            }

            if total_evidence >= self.min_evidence_magnitude:
                max_evidence = max(e_fake, e_real)
                if max_evidence > 0:
                    conflict_ratio = min(e_fake, e_real) / max_evidence
                    evidence_metrics["conflict_ratio"] = conflict_ratio

                    if conflict_ratio > self.conflict_threshold:
                        logger.info(
                            f"Early Stop Blocked: Evidential Conflict Ratio ({conflict_ratio:.2f})"
                        )
                        return StopDecision(
                            False,
                            StopReason.CONTINUE_ADVERSARIAL_CONFLICT,
                            current_score,
                            evidence_metrics=evidence_metrics
                        )

        # 6. Mathematical Bounds
        viable_denominator = weights_run + weights_pending

        if viable_denominator <= 0:
            return StopDecision(False, StopReason.CONTINUE_AMBIGUOUS, current_score)

        max_possible_score = (weighted_sum + (1.0 * weights_pending)) / viable_denominator
        min_possible_score = (weighted_sum + (0.0 * weights_pending)) / viable_denominator

        evidence_metrics["max_possible"] = max_possible_score
        evidence_metrics["min_possible"] = min_possible_score

        # 7. Locked REAL Check 
        if max_possible_score < self.fake_threshold and current_score < self.real_threshold:
            if not self.high_trust_tools.intersection(valid_tools.keys()):
                return StopDecision(
                    False,
                    StopReason.CONTINUE_SECURITY_REQUIRED,
                    current_score,
                    evidence_metrics=evidence_metrics
                )
            return StopDecision(
                True,
                StopReason.HALT_LOCKED_REAL,
                1.0 - current_score,
                evidence_metrics=evidence_metrics
            )

        # 8. Locked FAKE Check
        if min_possible_score > self.real_threshold and current_score > self.fake_threshold:
            if not self.high_trust_tools.intersection(valid_tools.keys()):
                return StopDecision(
                    False,
                    StopReason.CONTINUE_SECURITY_REQUIRED,
                    current_score,
                    evidence_metrics=evidence_metrics
                )
            return StopDecision(
                True,
                StopReason.HALT_LOCKED_FAKE,
                current_score,
                evidence_metrics=evidence_metrics
            )

        # 9. Default Continue
        return StopDecision(
            False,
            StopReason.CONTINUE_AMBIGUOUS,
            current_score,
            evidence_metrics=evidence_metrics
        )