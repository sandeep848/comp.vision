import os
import pandas as pd
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deepfake_robustness.configs import config

def generate_manifest():
    print("[Phase 2] Generating Compositional Manifest...")
    base_manifest = pd.read_csv(config.MANIFEST_PATH)
    
    # Operators
    operators = ["jpeg", "downscale", "motion_blur", "gaussian_blur", "gaussian_noise", "sharpen"]
    
    # 30 ordered pairs
    ordered_pairs = []
    for op1 in operators:
        for op2 in operators:
            if op1 != op2:
                ordered_pairs.append(f"{op1}->{op2}")
                
    # We will assign a pipeline_id to each row. 
    # To keep it balanced, we replicate the base manifest for different pipelines.
    # But wait, generating the full cartesian product might be huge.
    # Instead, let's assign a pipeline_id randomly to each row in the train split,
    # ensuring all pipelines are covered.
    
    new_rows = []
    for _, row in base_manifest.iterrows():
        split = row.get("split", "train")
        if split == "train":
            # Clean, 6 single operators, 30 ordered pairs (37 total configurations)
            # We can pick one randomly or duplicate. Let's just pick one randomly to keep dataset size same,
            # or duplicate it to give enough samples for each. Let's pick randomly.
            choices = ["clean"] + operators + ordered_pairs
            pipeline_id = random.choice(choices)
            r = row.copy()
            r["pipeline_id"] = pipeline_id
            new_rows.append(r)
        elif split == "val":
            # Validation: same distribution
            choices = ["clean"] + operators + ordered_pairs
            pipeline_id = random.choice(choices)
            r = row.copy()
            r["pipeline_id"] = pipeline_id
            new_rows.append(r)
        else:
            # Test: Held-out combinations of size >= 3, plus order-isolation pairs
            # Example held out: jpeg->downscale->gaussian_blur
            choices = ["jpeg->downscale->gaussian_blur", "gaussian_blur->jpeg->downscale", "order_isolation_test"]
            pipeline_id = random.choice(choices)
            r = row.copy()
            r["pipeline_id"] = pipeline_id
            new_rows.append(r)
            
    comp_df = pd.DataFrame(new_rows)
    out_path = config.PROJECT_ROOT / "compositional_manifest.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"Generated compositional manifest at {out_path} with {len(comp_df)} rows.")
    print("Pipeline distributions in Train:")
    print(comp_df[comp_df['split'] == 'train']['pipeline_id'].value_counts())

if __name__ == "__main__":
    generate_manifest()
