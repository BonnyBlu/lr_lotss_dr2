## This script is to merge all the temporary yaml files at the end of the batch run.

##################

### Imports ###

import yaml
import glob
from pathlib import Path
import os
import sys

##################

### Files and inputs ###
suffix = sys.argv[2].strip() if len(sys.argv) >= 3 else ""

if len(sys.argv) >= 2:
    base_path = Path(sys.argv[1]).resolve()
else:
    try:
        base_path = Path(__file__).resolve().parents[1]
    except NameError:
        base_path = Path.cwd()

os.chdir(base_path)

data_path = base_path / "data"
temp_dir = data_path / "tmp"
out_file = data_path / "outputs" / f"lr_outputs{suffix}.yml"
pattern = f"lr_outputs_*{suffix}.yml" if suffix else "lr_outputs_*.yml"

### Main code ###

merged_data = {}

# Read all temp files
for path in temp_dir.glob(pattern):
    with path.open() as f:
        data = yaml.safe_load(f)
        if data is None:
            continue
        job_id = path.stem.replace("lr_outputs_", "")
        merged_data[job_id] = data

# Write merged file
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open("w") as f:
    yaml.safe_dump(merged_data, f)

# Optional cleanup
for path in temp_dir.glob(pattern):
    path.unlink()

print(f"Merged {len(merged_data)} files into {out_file}")

##################