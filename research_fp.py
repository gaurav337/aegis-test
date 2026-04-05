import os
import sys
from pathlib import Path
from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.agent import ForensicAgent

def research_files(paths):
    config = AegisConfig()
    preprocessor = Preprocessor(config)
    agent = ForensicAgent(config)
    
    for path in paths:
        print(f"\n{'='*60}")
        print(f"RESEARCHING FILE: {path}")
        print(f"{'='*60}")
        
        try:
            prep_result = preprocessor.process_media(path)
            if not prep_result.has_face:
                print("No face detected.")
            
            # Record each tool result
            for event in agent.analyze(prep_result, media_path=str(path), generate_explanation=False):
                if event.event_type == "tool_complete":
                    tool = event.tool_name
                    data = event.data
                    # data = {"success": True, "real_prob": 0.5, "confidence": 0.0, "evidence_summary": ..., "error_msg": ...}
                    print(f"Tool: {tool:20} RealProb: {data.get('real_prob', 0.5):.4f} Conf: {data.get('confidence', 0.0):.4f} Summary: {data.get('evidence_summary', '')}")
                
                if event.event_type == "verdict":
                    print(f"\nFINAL VERDICT: {event.data.get('verdict')} (Authenticity: {event.data.get('real_prob'):.4f})")
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    files_to_check = [
        "/home/gaurav/Downloads/archive/Dataset/Validation/Real/real_16119.jpg",
        "/home/gaurav/Downloads/archive/Dataset/Validation/Real/real_10978.jpg"
    ]
    research_files(files_to_check)
