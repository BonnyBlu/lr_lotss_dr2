#!/usr/bin/env python3

import yaml
import sys

def flatten_dict(d, parent_key="", sep="_"):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Ensure correct usage 
if len(sys.argv) < 2:
    print("Usage: call_yaml.py <inputs.yml> [section] [keys...]")    # Print message if not enough sys.arg's
    sys.exit(1)

yaml_file = sys.argv[1]                                              # The first input is the yaml file
section = sys.argv[2] if len(sys.argv) > 2 else None                 # If the second input exists it is the section in the yaml file
keys = sys.argv[3:] if len(sys.argv) > 3 else None                   # If three+ inputs exists they are the exact keys to select

# Read YAML file
with open(yaml_file, "r") as file:                                   # Open the yaml file
    data = yaml.safe_load(file)

# Extract only the desired section
if section:
    data = data.get(section, {})                                     # Find the section

# Flatten the dictionary
flat_data = flatten_dict(data)                                       # Flatten the section to a list

# Filter only requested keys if specified
if keys:
    flat_data = {k: v for k, v in flat_data.items() if k in keys}    # Match up the keys and values

# Print shell export statements
for key, value in flat_data.items():
    print(f'export {key.upper()}="{value}"')                         # Convert keys to uppercase and export to shell


