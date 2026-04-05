"""Core data types for the Aegis-X system.

This module contains the unified interface and payload contracts for all internal
forensic tools and orchestration components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(init=False)
class ToolResult:
    """Standardized output payload from any forensic tool in Aegis-X."""
    
    tool_name: str
    success: bool
    real_prob: float
    confidence: float
    details: Dict[str, Any]
    error: bool
    error_msg: Optional[str]
    execution_time: float
    evidence_summary: str

    def __init__(
        self,
        tool_name: str,
        success: bool,
        real_prob: float = 0.0,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        error: bool = False,
        error_msg: Optional[str] = None,
        execution_time: float = 0.0,
        evidence_summary: str = "",
    ):
        self.tool_name = tool_name
        self.success = success

        # Canonical contract: continuous authenticity probability in [0, 1].
        self.real_prob = float(max(0.0, min(1.0, real_prob)))
            
        self.confidence = confidence
        self.details = details if details is not None else {}
        self.error = error
        self.error_msg = error_msg
        self.execution_time = execution_time
        self.evidence_summary = evidence_summary