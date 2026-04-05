"""Diagnose GPU tool outputs directly — bypasses the agent, subprocess proxy, and ensemble.
Runs each GPU tool on ONE known real and ONE known fake image to verify raw output."""

import os, sys, pickle, subprocess, tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ["AEGIS_MODEL_DIR"] = str(project_root / "models")

from core.config import AegisConfig
from utils.preprocessing import Preprocessor

GPU_TOOLS = ["run_freqnet", "run_univfd", "run_xception", "run_sbi"]

def get_one_image(dataset_root, subdir, label):
    p = Path(dataset_root) / subdir
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files = list(p.rglob(ext))
        if files:
            return files[0], label
    return None, None

def run_tool_subprocess(tool_name, input_data):
    """Run tool through subprocess exactly as SubprocessToolProxy does."""
    venv_gpu = project_root / ".venv_gpu" / "bin" / "python"
    worker = project_root / "core" / "subprocess_worker.py"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        # Strip frames_30fps like the proxy does
        slim = {k: v for k, v in input_data.items() if k != 'frames_30fps'}
        pickle.dump(slim, f)
        in_path = f.name
    
    out_path = in_path + ".out"
    env = os.environ.copy()
    env["AEGIS_MODEL_DIR"] = str(project_root / "models")
    
    try:
        result = subprocess.run(
            [str(venv_gpu), str(worker), tool_name, in_path],
            capture_output=True, text=True, timeout=120, env=env
        )
        
        if result.returncode != 0:
            print(f"  [{tool_name}] SUBPROCESS FAILED (rc={result.returncode})")
            print(f"    STDERR: {result.stderr[-500:]}")
            return None
        
        if not os.path.exists(out_path):
            print(f"  [{tool_name}] NO OUTPUT FILE")
            return None
        
        with open(out_path, 'rb') as f:
            tool_result = pickle.load(f)
        
        if isinstance(tool_result, Exception):
            print(f"  [{tool_name}] WORKER EXCEPTION: {tool_result}")
            return None
        
        return tool_result
    finally:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

def main():
    dataset_root = "/home/gaurav/Downloads/archive/Dataset/Test"
    
    real_file, real_label = get_one_image(dataset_root, "Real", 0)
    fake_file, fake_label = get_one_image(dataset_root, "Fake", 1)
    
    if not real_file or not fake_file:
        print("Could not find test images!")
        return
    
    print(f"REAL image: {real_file.name}")
    print(f"FAKE image: {fake_file.name}")
    print()
    
    config = AegisConfig()
    preprocessor = Preprocessor(config)
    
    for img_file, label, label_name in [(real_file, 0, "REAL"), (fake_file, 1, "FAKE")]:
        print(f"=== {label_name}: {img_file.name} ===")
        
        prep = preprocessor.process_media(str(img_file))
        print(f"  has_face={prep.has_face}, tracked_faces={len(prep.tracked_faces)}")
        if prep.tracked_faces:
            tf = prep.tracked_faces[0]
            print(f"  face_crop_224 shape={tf.face_crop_224.shape if tf.face_crop_224 is not None else None}")
            print(f"  face_crop_380 shape={tf.face_crop_380.shape if tf.face_crop_380 is not None else None}")
        
        input_data = {
            "media_path": str(img_file),
            "tracked_faces": prep.tracked_faces,
            "first_frame": prep.first_frame,
            "original_media_type": "image",
            "heuristic_flags": prep.heuristic_flags,
        }
        
        for tool_name in GPU_TOOLS:
            result = run_tool_subprocess(tool_name, input_data)
            if result is None:
                continue
            print(f"  [{tool_name}] real_prob={result.real_prob:.4f}, confidence={result.confidence:.2f}, "
                  f"success={result.success}, error={result.error}")
            if result.error_msg:
                print(f"    error_msg: {result.error_msg}")
            if result.evidence_summary:
                print(f"    evidence: {result.evidence_summary[:120]}")
            # Print key details
            for k in ("weights_loaded_ok", "boundary_detected", "skipped", "reason"):
                if k in result.details:
                    print(f"    {k}: {result.details[k]}")
        print()

if __name__ == "__main__":
    main()
