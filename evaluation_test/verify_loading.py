import os
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

if "AEGIS_MODEL_DIR" not in os.environ:
    os.environ["AEGIS_MODEL_DIR"] = str(project_root / "models")

from core.tools.freqnet_tool import FreqNetTool
from core.tools.univfd_tool import UnivFDTool
from core.tools.xception_tool import XceptionTool
from core.tools.sbi_tool import SBITool

def test_tool(tool_class):
    tool = tool_class()
    model = tool._load_model()
    
    total_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters()) if hasattr(model, "parameters") else 0
    
    # Actually move it to CUDA to verify memory footprint
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        allocated = torch.cuda.memory_allocated(0) / 1e6
        print(f"[{tool.tool_name}] Model params: {total_params}, Non-zero: {nonzero_params}")
        print(f"[{tool.tool_name}] CUDA memory allocated: {allocated:.1f}MB")
        # Clear exact memory allocated
        del model
        torch.cuda.empty_cache()
    else:
        print(f"[{tool.tool_name}] Model params: {total_params}, Non-zero: {nonzero_params} (CPU only)")

if __name__ == "__main__":
    print(f"AEGIS_MODEL_DIR = {os.environ.get('AEGIS_MODEL_DIR')}")
    test_tool(FreqNetTool)
    test_tool(UnivFDTool)
    test_tool(XceptionTool)
    test_tool(SBITool)
