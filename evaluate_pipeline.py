import os
import argparse
from pathlib import Path
from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.agent import ForensicAgent

try:
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    print("Missing required packages. Please install them by running:")
    print("source .venv_main/bin/activate")
    print("pip install pandas tqdm scikit-learn")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Aegis-X pipeline on a dataset.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--real_dir", type=str, default="real", help="Subdirectory name for real media.")
    parser.add_argument("--fake_dir", type=str, default="fake", help="Subdirectory name for fake media.")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Path to save the results CSV.")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    
    real_path = Path(args.dataset_root) / args.real_dir
    fake_path = Path(args.dataset_root) / args.fake_dir
    
    if not real_path.exists() or not fake_path.exists():
        print(f"Error: Could not find dataset directories. Ensure {real_path} and {fake_path} exist.")
        return
        
    media_files = []
    # Collect real files (label 0)
    for ext in ["*.mp4", "*.avi", "*.png", "*.jpg", "*.jpeg"]:
        media_files.extend([(f, 0) for f in real_path.rglob(ext)])
    
    # Collect fake files (label 1)
    for ext in ["*.mp4", "*.avi", "*.png", "*.jpg", "*.jpeg"]:
        media_files.extend([(f, 1) for f in fake_path.rglob(ext)])
        
    if not media_files:
        print("No media files found in the specified directories.")
        return
        
    print(f"Found {len([m for m in media_files if m[1] == 0])} real and {len([m for m in media_files if m[1] == 1])} fake files.")
    
    config = AegisConfig()
    preprocessor = Preprocessor(config)
    
    results = []
    y_true = []
    y_scores = []
    y_pred = []
    
    for file_path, label in tqdm(media_files, desc="Evaluating Pipeline"):
        try:
            # 1. Preprocess
            prep_result = preprocessor.process_media(str(file_path))
            if not prep_result.has_face:
                print(f"\nSkipping {file_path.name}: No faces detected.")
                continue
                
            # 2. Analyze
            agent = ForensicAgent(config)
            final_verdict = None
            final_score = 0.0
            
            # The agent.analyze method is a generator, so we iterate through its events
            for event in agent.analyze(prep_result, media_path=str(file_path)):
                if event.event_type == "verdict":
                    final_verdict = event.data.get("verdict")
                    final_score = event.data.get("score")
            
            # Label 1 = Fake, Label 0 = Real
            predicted_label = 1 if final_verdict == "FAKE" else 0
            
            results.append({
                "filename": file_path.name,
                "path": str(file_path),
                "true_label": label,
                "predicted_label": predicted_label,
                "score": final_score,
                "verdict": final_verdict
            })
            
            y_true.append(label)
            y_scores.append(1.0 - final_score)  # final_score is authenticity; ROC AUC expects fake prob = 1
            y_pred.append(predicted_label)
            
        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")
            continue
            
    # Calculate metrics
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float('nan') # E.g., if only one class is present
            
        print("\n" + "="*40)
        print("          EVALUATION RESULTS")
        print("="*40)
        print(f"Total processed: {len(y_true)}")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Precision:       {prec:.4f}")
        print(f"Recall:          {rec:.4f}")
        print(f"F1 Score:        {f1:.4f}")
        print(f"ROC AUC:         {auc:.4f}")
        print("="*40)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Detailed results saved to {args.output}")
    else:
        print("\nNo files were successfully processed.")

if __name__ == "__main__":
    evaluate()
