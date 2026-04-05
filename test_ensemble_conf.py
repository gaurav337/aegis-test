from core.data_types import ToolResult
from utils.ensemble import calculate_ensemble_score

tool_results = [
    ToolResult(tool_name="run_xception", success=True, real_prob=0.95, confidence=0.20),
    ToolResult(tool_name="run_sbi", success=True, real_prob=0.45, confidence=0.85),
]

res1 = calculate_ensemble_score(tool_results, return_metadata=True, use_confidence_weighting=False)
res2 = calculate_ensemble_score(tool_results, return_metadata=True, use_confidence_weighting=True)

print(f"Without Confidence Weighting: Fake Prob {res1['ensemble_score']*100:.1f}%")
print(f"With Confidence Weighting: Fake Prob {res2['ensemble_score']*100:.1f}%")
print(res2['weight_breakdown'])
