"""
Aegis-X Tool Diagnostics
========================
Run this to verify ALL CPU and GPU tools are working correctly:
  - Weights are loading from .pth files (not random init)
  - Scores/confidence are sensible
  - No fatal exceptions

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv_main/bin/python verify_tools.py
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv_main/bin/python verify_tools.py --image path/to/face.jpg
"""

import os
import sys
import argparse
from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.tools.registry import get_registry


def test_all_tools(test_image: str):
    if not os.path.exists(test_image):
        print(f"Error: image not found at '{test_image}'")
        return

    print("=" * 55)
    print("      AEGIS-X FULL TOOL DIAGNOSTICS")
    print("=" * 55)
    print(f"Image: {test_image}\n")

    # ── Preprocessing ──────────────────────────────────────
    print("[✔] Running Preprocessor...")
    config = AegisConfig()
    prep = Preprocessor(config)
    prep_result = prep.process_media(test_image)

    if not prep_result.has_face:
        print("[✘] No faces detected in image. All tools require a face — aborting.")
        return

    print(f"    Faces tracked : {len(prep_result.tracked_faces)}")
    print(f"    Media type    : {prep_result.original_media_type}")
    print(f"    Frames (30fps): {len(prep_result.frames_30fps)}")

    input_data = {
        "tracked_faces":      prep_result.tracked_faces,
        "frames_30fps":       prep_result.frames_30fps,
        "media_path":         test_image,
        "original_media_type": prep_result.original_media_type,
    }

    registry = get_registry()
    cpu_tools = registry.get_cpu_tools()
    gpu_tools = registry.get_gpu_tools()

    # ── Helper ─────────────────────────────────────────────
    def run_tool(tool_name: str, skip_reason: str = None):
        print(f"\n  [{tool_name.upper()}]")
        if skip_reason:
            print(f"   -> Skipped: {skip_reason}")
            return

        try:
            tool = registry.get_tool(tool_name)
            res  = tool.execute(input_data)

            status = "✔ OK" if res.success else "✘ FAILED"
            print(f"   -> Status    : {status}")
            if not res.success:
                print(f"   -> Error     : {res.error_msg}")
            print(f"   -> Real Prob : {res.real_prob:.3f}")
            print(f"   -> Confidence: {res.confidence:.3f}")
            print(f"   -> Summary   : {res.evidence_summary}")
        except Exception as exc:
            print(f"   -> FATAL EXCEPTION: {exc}")

    # ── CPU Tools ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CPU TOOLS")
    print("=" * 55)

    for tool_name in cpu_tools:
        skip = None
        if tool_name == "run_rppg" and prep_result.original_media_type == "image":
            skip = "rPPG requires video — this is a static image."
        run_tool(tool_name, skip)

    # ── GPU Tools ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GPU TOOLS")
    print("=" * 55)

    for tool_name in gpu_tools:
        run_tool(tool_name)

    print("\n" + "=" * 55)
    print("  ALL TESTS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis-X Tool Diagnostics")
    parser.add_argument(
        "--image", "-i",
        default="downloads/web_uploads/fake_1007.jpg",
        help="Path to the test image (must contain a face)."
    )
    args = parser.parse_args()
    test_all_tools(args.image)
