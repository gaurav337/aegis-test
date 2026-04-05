#!/usr/bin/env python3
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Aegis Imports
from core.tools.freqnet.preprocessor import DCTPreprocessor
from core.tools.freqnet.fad_hook import FADHook

def compute_calibration(dataset_path, output_path="calibration/freqnet_fad_baseline.pt"):
    print(f"Starting FreqNet Calibration on: {dataset_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dct_prep = DCTPreprocessor().to(device)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset_root = Path(dataset_path)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(list(dataset_root.rglob(ext)))
        
    if not image_files:
        print(f"Error: No images found in {dataset_path}")
        return

    print(f"Found {len(image_files)} images. Analyzing...")
    
    all_base, all_mid, all_high = [], [], []
    
    for img_path in tqdm(image_files[:500]): # Limit to 500 for speed
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            
            # Extract patches (mirroring FreqNetTool logic)
            h, w = img_np.shape[:2]
            patch_size = 224
            if h > patch_size and w > patch_size:
                # Take center patch
                y, x = (h - patch_size) // 2, (w - patch_size) // 2
                patch = img_np[y:y+patch_size, x:x+patch_size]
            else:
                patch = cv2.resize(img_np, (patch_size, patch_size))
                
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = tensor.to(device)
            
            hook = FADHook()
            hook.register(dct_prep._dct_conv)
            _ = dct_prep(tensor)
            hook.remove()
            
            band = hook.analyze()
            if band and band.total_energy > 0:
                all_base.append(band.base_ratio)
                all_mid.append(band.mid_ratio)
                all_high.append(band.high_ratio)
                
        except Exception as e:
            continue

    if not all_base:
        print("Error: No valid frequency data extracted.")
        return

    calibration_data = {
        'mean_base': float(np.mean(all_base)),
        'std_base': float(np.std(all_base)),
        'mean_mid': float(np.mean(all_mid)),
        'std_mid': float(np.std(all_mid)),
        'mean_high': float(np.mean(all_high)),
        'std_high': float(np.std(all_high)),
        'num_samples': len(all_base)
    }

    torch.save(calibration_data, output_path)
    print(f"\n✅ Calibration complete! Saved to {output_path}")
    print(f"Samples processed: {len(all_base)}")
    print(f"Base Mean: {calibration_data['mean_base']:.4f}")
    print(f"Mid Mean:  {calibration_data['mean_mid']:.4f}")
    print(f"High Mean: {calibration_data['mean_high']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/home/gaurav/Downloads/archive/Dataset/Validation/Real")
    parser.add_argument("--output", type=str, default="calibration/freqnet_fad_baseline.pt")
    args = parser.parse_args()
    
    compute_calibration(args.dataset, args.output)
