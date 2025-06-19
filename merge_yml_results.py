## This script is to merge all the temporary yaml files at the end of the batch run.

##################

### Imports ###

import yaml
import glob
import os
import sys

##################

### Files and inputs ###
suffix = sys.argv[2] if len(sys.argv) > 1 else ""
suffix = suffix.strip()

dir = sys.argv[1]                               # Working directory to change to and run the code from
os.chdir(dir)                                   # Move to working/data directory (should be bound to container)


try:
    BASEPATH = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(BASEPATH, "data")
except NameError:
    if os.path.exists("data"):
        BASEPATH = "."
        data_path = os.path.join(BASEPATH, "data")
    else:
        BASEPATH = os.getcwd()
        data_path = os.path.join(BASEPATH, "..", "..", "data")

temp_dir = os.path.join(data_path,"outputs", "tmp")
out_file = os.path.join(data_path, "outputs", f"lr_outputs{suffix}.yml")

### Main code ###

merged_data = {}

for path in glob.glob(os.path.join(temp_dir, "lr_outputs_*.yml")):
    with open(path) as f:
        data = yaml.safe_load(f)
        if data is None:
            continue
        job_id = os.path.splitext(os.path.basename(path))[0].replace("lr_outputs_", "")
        merged_data[job_id] = data

# Write merged file
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w") as f:
    yaml.safe_dump(merged_data, f)

# Optional cleanup
for path in glob.glob(os.path.join(temp_dir, "lr_outputs_*.yml")):
    os.remove(path)

print(f"Merged {len(merged_data)} files into {out_file}")

##################