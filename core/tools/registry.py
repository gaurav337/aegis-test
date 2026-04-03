"""
Tool Registry: single source of truth for tool metadata and execution.

Every tool in the Aegis-X pipeline is registered here with its weight,
category, trust tier, AND runtime instance. This prevents magic strings
and ensures weight calculations are consistent across the system.
"""

import importlib
import logging
import time
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum, auto

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Tool Metadata & Categories
# ──────────────────────────────────────────────────────────────

class ToolCategory(Enum):
    """Distinct analytical categories for diversity checks."""
    FREQUENCY = auto()     # Pixel/frequency artifact analysis
    SEMANTIC = auto()      # High-level semantic/CLIP analysis
    GEOMETRIC = auto()     # Face geometry / landmark analysis
    PROVENANCE = auto()    # C2PA / metadata verification
    GENERATIVE = auto()    # SBI / GAN fingerprint detection
    BIOLOGICAL = auto()    # RPPG / corneal / biological signals


@dataclass(frozen=True)
class ToolSpec:
    """
    Immutable specification for a pipeline tool.

    Attributes
    ----------
    name : str
        Canonical tool function name (e.g., "sbi", "freqnet").
    weight : float
        Base weight in the ensemble. Must be > 0.
    category : ToolCategory
        Analytical category for diversity gating.
    trust_tier : int
        1 = Low (cheap, artifact-prone to FP)
        2 = Medium (balanced accuracy/cost)
        3 = High (expensive, high accuracy — must-run for safety)
    """
    name: str
    weight: float
    category: ToolCategory
    trust_tier: int

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"Weight must be > 0, got {self.weight}")
        if self.trust_tier not in (1, 2, 3):
            raise ValueError(f"Trust tier must be 1, 2, or 3, got {self.trust_tier}")


# ──────────────────────────────────────────────────────────────
# Tool Manifest: (label, module_path, class_name, weight, category, tier)
# Single source of truth for what tools exist + their metadata
# ──────────────────────────────────────────────────────────────
# In core/tools/registry.py, update _TOOL_MANIFEST:

_TOOL_MANIFEST = [
    ("check_c2pa",       "core.tools.c2pa_tool",         "C2PATool",         0.05, ToolCategory.PROVENANCE,  1),
    ("run_dct",          "core.tools.dct_tool",          "DCTTool",          0.07, ToolCategory.FREQUENCY,   2),
    ("run_rppg",         "core.tools.rppg_tool",         "RPPGTool",         0.06, ToolCategory.BIOLOGICAL,  2),
    ("run_geometry",     "core.tools.geometry_tool",     "GeometryTool",     0.18, ToolCategory.GEOMETRIC,   3),
    ("run_illumination", "core.tools.illumination_tool", "IlluminationTool", 0.05, ToolCategory.FREQUENCY,   1),
    ("run_corneal",      "core.tools.corneal_tool",      "CornealTool",      0.07, ToolCategory.BIOLOGICAL,  2),
    ("run_univfd",       "core.tools.univfd_tool",       "UnivFDTool",       0.20, ToolCategory.SEMANTIC,    3),
    ("run_xception",     "core.tools.xception_tool",     "XceptionTool",     0.15, ToolCategory.SEMANTIC,    2),
    ("run_sbi",          "core.tools.sbi_tool",          "SBITool",          0.20, ToolCategory.GENERATIVE,  3),
    ("run_freqnet",      "core.tools.freqnet_tool",      "FreqNetTool",      0.09, ToolCategory.FREQUENCY,   1),
]

# Total: 1.00 ✓


def _build_metadata_registry() -> Dict[str, ToolSpec]:
    """Build the metadata registry from manifest."""
    registry = {}
    for label, _, _, weight, category, tier in _TOOL_MANIFEST:
        registry[label] = ToolSpec(
            name=label,
            weight=weight,
            category=category,
            trust_tier=tier
        )
    
    # Validate weights sum to ~1.0
    total = sum(spec.weight for spec in registry.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning(
            f"Tool weights sum to {total}, not 1.0. "
            f"This may affect ensemble calculations."
        )
    
    return registry


# ──────────────────────────────────────────────────────────────
# Runtime Tool Registry
# ──────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all forensic tools.
    
    Responsibilities:
      - Import and instantiate tools with fault isolation
      - Dispatch execute calls by tool name
      - Track tool health (registered vs failed)
      - Expose tool metadata (weights, trust tiers) for EarlyStoppingController
      - Clear GPU memory between tool executions
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseForensicTool] = {}
        self.failed_tools: Dict[str, str] = {}
        self._execution_counts: Dict[str, int] = {}
        self._total_execution_time: Dict[str, float] = {}
        
        # NEW: Metadata registry for EarlyStoppingController
        self._metadata: Dict[str, ToolSpec] = _build_metadata_registry()
        
        self._register_all()
        
        registered = list(self.tools.keys())
        failed = list(self.failed_tools.keys())
        logger.info(
            f"ToolRegistry initialized: {len(registered)} active, "
            f"{len(failed)} failed"
        )
        if registered:
            logger.info(f"  Active: {registered}")
        if failed:
            logger.warning(f"  Failed: {self.failed_tools}")
    
    def _register_all(self):
        """Import and instantiate all tools. Each tool is isolated —
        one failure does not affect others."""
        
        from core.subprocess_proxy import SubprocessToolProxy
        GPU_TOOLS = {"run_univfd", "run_xception", "run_sbi", "run_freqnet"}
        
        for label, module_path, class_name, _, _, _ in _TOOL_MANIFEST:
            try:
                if label in GPU_TOOLS:
                    # Proxy isolated GPU tools
                    instance = SubprocessToolProxy(label)
                    
                    # Verify worker can initialize
                    health_check = instance.health_check()
                    if not health_check.success:
                        self.failed_tools[label] = "Worker setup failed: " + str(health_check.error_msg)
                        continue
                        
                    tool_name = instance.tool_name
                    
                    self.tools[tool_name] = instance
                    self._execution_counts[tool_name] = 0
                    self._total_execution_time[tool_name] = 0.0
                    logger.debug(f"  Registered Proxy: {tool_name} ({class_name})")
                    continue
                
                # Phase 1: Import the module
                module = importlib.import_module(module_path)
                
                # Phase 2: Get the class
                cls = getattr(module, class_name)
                
                # Phase 3: Instantiate
                instance = cls()
                
                # Phase 4: Verify it has the required interface
                if not isinstance(instance, BaseForensicTool):
                    raise TypeError(
                        f"{class_name} does not extend BaseForensicTool"
                    )
                if not hasattr(instance, 'tool_name'):
                    raise AttributeError(
                        f"{class_name} missing tool_name property"
                    )
                
                tool_name = instance.tool_name
                
                # Phase 5: Check for name collisions
                if tool_name in self.tools:
                    raise ValueError(
                        f"Duplicate tool_name '{tool_name}': "
                        f"{class_name} collides with "
                        f"{type(self.tools[tool_name]).__name__}"
                    )
                
                # Phase 6: Try setup with fault isolation
                if hasattr(instance, 'setup') and callable(instance.setup):
                    try:
                        instance.setup()
                    except Exception as setup_err:
                        logger.warning(
                            f"{class_name}.setup() failed: {setup_err}. "
                            f"Tool registered but may be degraded."
                        )
                
                self.tools[tool_name] = instance
                self._execution_counts[tool_name] = 0
                self._total_execution_time[tool_name] = 0.0
                logger.debug(f"  Registered: {tool_name} ({class_name})")
                
            except ImportError as e:
                self.failed_tools[label] = f"ImportError: {e}"
                logger.error(f"  Cannot import {label}: {e}")
            except Exception as e:
                self.failed_tools[label] = f"{type(e).__name__}: {e}"
                logger.error(f"  Failed to register {label}: {e}")
    
    # ──────────────────────────────────────────────────────────
    # NEW: Metadata Access Methods for EarlyStoppingController
    # ──────────────────────────────────────────────────────────
    
    def get_tool_spec(self, name: str) -> Optional[ToolSpec]:
        """Get ToolSpec metadata for a tool by name."""
        return self._metadata.get(name)
    
    def get_all_tool_specs(self) -> Dict[str, ToolSpec]:
        """Get all ToolSpec metadata. For EarlyStoppingController init."""
        return dict(self._metadata)
    
    def get_high_trust_tools(self) -> List[str]:
        """Get list of high-trust (tier 3) tool names."""
        return [
            name for name, spec in self._metadata.items()
            if spec.trust_tier == 3
        ]
    
    def get_viable_pending_tools(self, completed_tools: List[str]) -> List[str]:
        """
        Get list of tools that are registered AND not yet completed.
        Excludes failed tools automatically.
        
        This is the CRITICAL method for EarlyStoppingController integration.
        """
        viable = []
        for name in self.tools.keys():
            if name not in completed_tools:
                viable.append(name)
        return viable
    
    def get_total_system_weight(self) -> float:
        """Get total weight of all tools in metadata registry."""
        return sum(spec.weight for spec in self._metadata.values())
    
    # ──────────────────────────────────────────────────────────
    # Execution Methods (unchanged)
    # ──────────────────────────────────────────────────────────
    
    def execute_tool(self, name: str, input_data: dict) -> ToolResult:
        """
        Execute a tool by name.
        
        Handles:
          - Tool not found → error ToolResult
          - Tool crashes → error ToolResult (never propagates exceptions)
          - GPU OOM → clear cache + retry once
          - VRAM cleanup after GPU tools
        """
        tool = self.tools.get(name)
        
        if tool is None:
            # Check if it was a failed registration
            if name in self.failed_tools:
                reason = self.failed_tools[name]
                msg = f"Tool '{name}' failed to register: {reason}"
            else:
                available = list(self.tools.keys())
                msg = f"Tool '{name}' not found. Available: {available}"
            
            return ToolResult(
                tool_name=name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg=msg,
                execution_time=0.0,
                evidence_summary=msg
            )
        
        # Detect GPU tools
        is_gpu = getattr(tool, 'requires_gpu', False)
        start = time.perf_counter()
        
        try:
            result = tool.execute(input_data)
            
            elapsed = time.perf_counter() - start
            self._execution_counts[name] += 1
            self._total_execution_time[name] += elapsed
            
            return result
            
        except Exception as e:
            if HAS_TORCH and isinstance(e, torch.cuda.OutOfMemoryError):
                logger.error(f"{name}: CUDA OOM — clearing cache and retrying")
                torch.cuda.empty_cache()
                
                try:
                    result = tool.execute(input_data)
                    elapsed = time.perf_counter() - start
                    self._execution_counts[name] += 1
                    self._total_execution_time[name] += elapsed
                    return result
                except Exception as retry_err:
                    return ToolResult(
                        tool_name=name,
                        success=False,
                        score=0.0,
                        confidence=0.0,
                        details={},
                        error=True,
                        error_msg=f"OOM even after cache clear: {retry_err}",
                        execution_time=time.perf_counter() - start,
                        evidence_summary=f"OOM even after cache clear: {retry_err}"
                    )
            # Generic execution failure:
            logger.error(f"{name} execution failed: {e}", exc_info=True)
                
            # Fallthrough for Generic exception is handled above!
            logger.error(f"{name} execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_name=name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg=f"Execution failed: {type(e).__name__}: {e}",
                execution_time=time.perf_counter() - start,
                evidence_summary=f"Execution failed: {type(e).__name__}: {e}"
            )
            
        finally:
            # Prevent VRAM fragmentation between tool calls
            if is_gpu and HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_tool_names(self) -> List[str]:
        """Return names of all successfully registered tools."""
        return list(self.tools.keys())
        
    def get_tool(self, name: str) -> Optional[BaseForensicTool]:
        """Get a specific tool instance by name."""
        return self.tools.get(name)

    def get_cpu_tools(self) -> List[str]:
        """Return names of all CPU-bound tools."""
        return [name for name, tool in self.tools.items() if not getattr(tool, 'requires_gpu', False)]

    def get_gpu_tools(self) -> List[str]:
        """Return names of all GPU-bound tools."""
        return [name for name, tool in self.tools.items() if getattr(tool, 'requires_gpu', False)]
    
    def get_health_report(self) -> dict:
        """Full health report for diagnostics."""
        return {
            "active_tools": list(self.tools.keys()),
            "failed_tools": dict(self.failed_tools),
            "active_count": len(self.tools),
            "failed_count": len(self.failed_tools),
            "execution_stats": {
                name: {
                    "calls": self._execution_counts.get(name, 0),
                    "total_time": round(
                        self._total_execution_time.get(name, 0), 3
                    ),
                }
                for name in self.tools
            },
        }
    
    def shutdown(self):
        """Clean shutdown — release all tool resources."""
        for name, tool in self.tools.items():
            try:
                if hasattr(tool, 'cleanup'):
                    tool.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {name}: {e}")
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.tools.clear()
        logger.info("ToolRegistry shut down")


# ──────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────
_registry: Optional[ToolRegistry] = None
_lock = threading.Lock()


def get_registry() -> ToolRegistry:
    """Get or create the global ToolRegistry singleton."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Destroy and reset the singleton. For testing only."""
    global _registry
    with _lock:
        if _registry is not None:
            _registry.shutdown()
        _registry = None