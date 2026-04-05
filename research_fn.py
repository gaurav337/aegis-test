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
            # Disable LLM explanation for speed
            for event in agent.analyze(prep_result, media_path=str(path), generate_explanation=False):
                if event.event_type == "tool_complete":
                    print(f"Tool: {event.tool_name:20} RealProb: {event.data.get('real_prob', 0.5):.4f} Conf: {event.data.get('confidence', 0.0):.4f}")
                if event.event_type == "verdict":
                    print(f"\nFINAL VERDICT: {event.data.get('verdict')} (Authenticity: {event.data.get('real_prob'):.4f})")
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    # Check a couple of "Missed Fakes"
    files_to_check = [
        "/home/gaurav/Downloads/archive/Dataset/Validation/Fake/fake_3326.jpg",
        "/home/gaurav/Downloads/archive/Dataset/Validation/Fake/fake_6093.jpg"
    ]
    research_files(files_to_check)
