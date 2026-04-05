import logging
logging.basicConfig(level=logging.INFO)
from core.data_types import ToolResult
from utils.ensemble import calculate_ensemble_score

tool_results = [
    ToolResult(tool_name="run_dct", success=True, real_prob=0.92, confidence=0.95),
    ToolResult(tool_name="run_geometry", success=True, real_prob=0.84, confidence=0.80),
    ToolResult(tool_name="run_freqnet", success=True, real_prob=1.00, confidence=0.95),
    ToolResult(tool_name="run_univfd", success=True, real_prob=0.57, confidence=0.52),
    ToolResult(tool_name="run_xception", success=True, real_prob=0.51, confidence=0.41),
    ToolResult(tool_name="run_sbi", success=True, real_prob=0.59, confidence=0.45),
]

res = calculate_ensemble_score(tool_results, return_metadata=True)
print(f"Final: {res['ensemble_score'] * 100:.1f}%")
