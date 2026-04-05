#!/usr/bin/env python3
"""
Aegis-X Specialized Evaluation Script (evaluate.py)
--------------------------------------------------
Performs a balanced 50/50 evaluation on 100 images from the Test dataset.
Results are stored in the 'evaluation/' folder.

Usage:
    ./.venv_main/bin/python evaluate.py
"""

import os
import time
import random
import argparse
import json
from datetime import datetime
from pathlib import Path

# Project Imports
from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.agent import ForensicAgent

# Optional UI Imports
try:
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

def setup_args():
    parser = argparse.ArgumentParser(description="Evaluate Aegis-X on a balanced 100-sample test set.")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="/home/gaurav/Downloads/archive/Dataset/Validation",
        help="Path to the dataset root (contains Real/Fake subfolders)"
    )
    parser.add_argument("--num_total", type=int, default=100, help="Total number of images to test (50 real / 50 fake)")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to store results")
    return parser.parse_args()

def evaluate():
    args = setup_args()
    
    # 1. Setup Directories
    dataset_root = Path(args.dataset_root)
    real_path = dataset_root / "Real"
    fake_path = dataset_root / "Fake"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not real_path.exists() or not fake_path.exists():
        print(f"Error: Could not find dataset subfolders in {dataset_root}")
        print("Expected structure: 'Real/' and 'Fake/' subdirectories.")
        return

    # 2. Balanced Selection (50/50)
    num_per_class = args.num_total // 2
    
    real_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.mp4"]:
        real_files.extend([(f, 0) for f in real_path.rglob(ext)])
        
    fake_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.mp4"]:
        fake_files.extend([(f, 1) for f in fake_path.rglob(ext)])
        
    if len(real_files) < num_per_class or len(fake_files) < num_per_class:
        print(f"Warning: Not enough files for a {num_per_class}/{num_per_class} split.")
        num_per_class = min(len(real_files), len(fake_files))
        print(f"Adjusting to {num_per_class}/{num_per_class} (Total: {num_per_class * 2})")

    sampled_real = random.sample(real_files, num_per_class)
    sampled_fake = random.sample(fake_files, num_per_class)
    
    test_set = sampled_real + sampled_fake
    random.shuffle(test_set)
    
    print("-" * 60)
    print(f"  AEGIS-X EVALUATION START")
    print(f"  Dataset: {dataset_root}")
    print(f"  Samples: {len(test_set)} (Balanced 50/50)")
    print(f"  Results Folder: {output_dir.absolute()}")
    print("-" * 60)

    # 3. Pipeline Initialization
    config = AegisConfig()
    preprocessor = Preprocessor(config)
    agent = ForensicAgent(config)
    
    results = []
    y_true, y_pred, y_scores = [], [], []

    # 4. Main Evaluation Loop
    iterator = tqdm(test_set, desc="Analyzing Media") if HAS_METRICS else test_set
    
    for file_path, true_label in iterator:
        try:
            # Preprocess
            prep_result = preprocessor.process_media(str(file_path))
            
            # Analyze
            final_verdict = "INCONCLUSIVE"
            final_score = 0.5
            
            # Iterate through agent events (mirrors run_web logic)
            for event in agent.analyze(prep_result, media_path=str(file_path), generate_explanation=False):
                if event.event_type == "verdict":
                    final_verdict = event.data.get("verdict")
                    final_score = event.data.get("real_prob")
            
            # Mapping: 1 = FAKE, 0 = REAL
            # final_score in Aegis is AUTHENTICITY (1.0 = Real, 0.0 = Fake)
            predicted_label = 1 if final_verdict == "FAKE" else 0
            
            # Track for metrics
            y_true.append(true_label)
            y_pred.append(predicted_label)
            y_scores.append(1.0 - final_score) # Probability of being Fake
            
            results.append({
                "timestamp": datetime.now().isoformat(),
                "filename": file_path.name,
                "path": str(file_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "authenticity_score": final_score,
                "verdict": final_verdict,
                "matched": (predicted_label == true_label)
            })
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {file_path.name}: {e}")
            continue

    # 5. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_results_{timestamp}.csv"
    metrics_file = output_dir / f"evaluation_metrics_{timestamp}.json"
    
    if HAS_METRICS:
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            "timestamp": timestamp,
            "total_samples": len(y_true),
            "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print("\n" + "="*60)
        print("                FINAL EVALUATION SUMMARY")
        print("="*60)
        print(f" Accuracy:  {acc*100:.2f}%")
        print(f" Precision: {prec*100:.2f}%")
        print(f" Recall:    {rec*100:.2f}%")
        print(f" F1 Score:  {f1:.4f}")
        print(f" ROC AUC:   {auc:.4f}")
        print("-" * 60)
        print(f" [TP: {tp}] [TN: {tn}] [FP: {fp}] [FN: {fn}]")
        print("-" * 60)
        print(f" Saved results to: {results_file}")
        print(f" Saved metrics to: {metrics_file}")
    else:
        # Fallback to simple JSON if pandas/sklearn missing
        with open(results_file.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {results_file.with_suffix('.json')}")

if __name__ == "__main__":
    evaluate()
