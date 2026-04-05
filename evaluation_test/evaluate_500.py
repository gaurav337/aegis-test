import os
import random
import argparse
from pathlib import Path

# Add project root to sys.path so it can find core, utils, etc.
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Ensure model directory is resolved relative to the project root, not cwd
if "AEGIS_MODEL_DIR" not in os.environ:
    os.environ["AEGIS_MODEL_DIR"] = str(project_root / "models")

from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.agent import ForensicAgent
try:
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
except ImportError:
    print("Missing required packages. Please install them by running:")
    print("source .venv_main/bin/activate")
    print("pip install pandas tqdm scikit-learn")
    import sys
    sys.exit(1)

def evaluate(dataset_root, real_dir="Real", fake_dir="Fake", num_samples=200):
    real_path = Path(dataset_root) / real_dir
    fake_path = Path(dataset_root) / fake_dir
    
    if not real_path.exists() or not fake_path.exists():
        print(f"Error: Could not find dataset directories. Ensure {real_path} and {fake_path} exist.")
        return
        
    # Collect real files (label 0)
    real_files = []
    for ext in ["*.mp4", "*.avi", "*.png", "*.jpg", "*.jpeg"]:
        real_files.extend([(f, 0) for f in real_path.rglob(ext)])
        
    # Collect fake files (label 1)
    fake_files = []
    for ext in ["*.mp4", "*.avi", "*.png", "*.jpg", "*.jpeg"]:
        fake_files.extend([(f, 1) for f in fake_path.rglob(ext)])
        
    if not real_files and not fake_files:
        print("No media files found in the specified directories.")
        return
        
    print(f"Found {len(real_files)} total real files and {len(fake_files)} total fake files.")
    
    # Randomly sample num_samples from each (or less if not enough files)
    num_real_samples = min(num_samples, len(real_files))
    num_fake_samples = min(num_samples, len(fake_files))
    
    sampled_real = random.sample(real_files, num_real_samples)
    sampled_fake = random.sample(fake_files, num_fake_samples)
    
    media_files = sampled_real + sampled_fake
    random.shuffle(media_files)
    
    print(f"Sampled {len(sampled_real)} real and {len(sampled_fake)} fake files for evaluation.")
    
    config = AegisConfig()
    config.agent.use_openrouter = True
    
    # Initialize Preprocessor. 
    preprocessor = Preprocessor(config)
    
    results = []
    y_true = []
    y_scores = []  # Probability of being FAKE
    y_pred = []
    
    for file_path, label in tqdm(media_files, desc="Evaluating Pipeline"):
        try:
            
            # 1. Preprocess
            prep_result = preprocessor.process_media(str(file_path))
            
            # 2. Analyze
            agent = ForensicAgent(config)
            final_verdict = None
            final_score = 0.0
            
            for event in agent.analyze(prep_result, media_path=str(file_path), generate_explanation=False):
                if event.event_type == "tool_complete":
                    t_name = event.tool_name or "?"
                    t_score = event.data.get("score", "?")
                    t_ok = event.data.get("success", False)
                    t_err = event.data.get("error_msg", None)
                    status = f"score={t_score:.3f}" if isinstance(t_score, float) else f"score={t_score}"
                    if not t_ok:
                        status += f" ERROR: {t_err}"
                    print(f"    [{t_name}] {status}", flush=True)
                if event.event_type == "verdict":
                    final_verdict = event.data.get("verdict")
                    final_score = event.data.get("score")
            
            # Label 1 = Fake, Label 0 = Real
            predicted_label = 1 if final_verdict == "FAKE" else 0
            
            true_label_str = 'Fake' if label == 1 else 'Real'
            pred_label_str = 'Fake' if predicted_label == 1 else 'Real'
            marker = "✓" if predicted_label == label else "✗"
            result_str = f"[{marker}] {file_path.name}: Predicted={pred_label_str}, True={true_label_str}, Score={final_score}"
            print(result_str, flush=True)
            
            results.append({
                "filename": file_path.name,
                "path": str(file_path),
                "true_label": label,
                "predicted_label": predicted_label,
                "score": final_score,
                "verdict": final_verdict
            })
            
            y_true.append(label)
            
            # final_score is authenticity; ROC AUC expects fake prob = 1
            y_scores.append(1.0 - final_score)
            y_pred.append(predicted_label)
            
        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")
            continue
            
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float('nan')
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        print("\n" + "="*50)
        print("          EVALUATION RESULTS")
        print("="*50)
        print(f"Total processed: {len(y_true)}")
        print(f"True Positives (TP): {tp}  (Fake predicted as Fake)")
        print(f"True Negatives (TN): {tn}  (Real predicted as Real)")
        print(f"False Positives (FP): {fp} (Real predicted as Fake)")
        print(f"False Negatives (FN): {fn} (Fake predicted as Real)")
        print("-" * 50)
        print(f"Accuracy:        {acc:.4f}")
        print(f"Precision:       {prec:.4f}")
        print(f"Recall:          {rec:.4f}")
        print(f"F1 Score:        {f1:.4f}")
        print(f"ROC AUC:         {auc:.4f}")
        print("="*50)
        
        output_dir = Path(__file__).resolve().parent
        results_csv = output_dir / "evaluation_results.csv"
        metrics_csv = output_dir / "evaluation_metrics.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(results_csv, index=False)
        
        metrics_df = pd.DataFrame([{
            "Total Processed": len(y_true),
            "True Positives (TP)": tp,
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc
        }])
        metrics_df.to_csv(metrics_csv, index=False)
        
        print(f"Detailed results saved to {results_csv}")
        print(f"Metrics saved to {metrics_csv}")
    else:
        print("\nNo files were successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Aegis-X on a sample dataset.")
    parser.add_argument("--dataset_root", type=str, default="/home/gaurav/Downloads/archive/Dataset/Test", help="Path to the dataset root directory.")
    parser.add_argument("--real_dir", type=str, default="Real", help="Subdirectory name for real media.")
    parser.add_argument("--fake_dir", type=str, default="Fake", help="Subdirectory name for fake media.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to take from each class.")
    
    args = parser.parse_args()
    
    evaluate(
        dataset_root=args.dataset_root, 
        real_dir=args.real_dir, 
        fake_dir=args.fake_dir, 
        num_samples=args.num_samples
    )
