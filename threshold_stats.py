## This script is for calculating the averages of the thresholds in each .yml file and writing the results to that file. ##

import sys
import yaml
import numpy as np
import scipy.stats
from pathlib import Path

##################################
## Check command-line arguments
##################################

if len(sys.argv) != 3:
    print("Usage: python3 threshold_stats_extractor.py <input_file.yml> <suffix>")
    sys.exit(1)

yml_file = sys.argv[1]
suffix = sys.argv[2]

input_path = Path(yml_file)
if not input_path.exists():
    print(f"Error: File '{input_path}' not found.")
    sys.exit(1)

##################################
## Read YAML and extract thresholds
##################################

with open(input_path, 'r') as f:
    data = yaml.safe_load(f)

thresholds = []
total_entries = 0
missing_entries = 0

for key, val in data.items():
    total_entries += 1
    logs = val.get('logs', [])
    found = False
    for log in logs:
        if 'threshold' in log:
            try:
                thresholds.append(float(log['threshold']))
                found = True
                break
            except (ValueError, TypeError):
                continue
    if not found:
        missing_entries += 1

##################################
## Compute statistics with native Python types and YAML-safe values
##################################

def safe_float(val):
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    except Exception:
        return None

stats = {}
if thresholds:
    arr = np.array(thresholds)
    stats['count'] = int(len(arr))
    stats['mean'] = safe_float(np.mean(arr))
    stats['median'] = safe_float(np.median(arr))
    stats['std_dev'] = safe_float(np.std(arr, ddof=1)) if len(arr) > 1 else None
    stats['sem'] = safe_float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None
    stats['min'] = safe_float(np.min(arr))
    stats['max'] = safe_float(np.max(arr))
    stats['25th_percentile'] = safe_float(np.percentile(arr, 25))
    stats['75th_percentile'] = safe_float(np.percentile(arr, 75))

    if len(arr) > 1:
        t_score = scipy.stats.t.ppf(0.975, df=len(arr) - 1)
        sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
        ci_range = t_score * sem
        ci_lower = np.mean(arr) - ci_range
        ci_upper = np.mean(arr) + ci_range
        stats['95%_CI_lower'] = safe_float(ci_lower)
        stats['95%_CI_upper'] = safe_float(ci_upper)
    else:
        stats['95%_CI_lower'] = None
        stats['95%_CI_upper'] = None

##################################
## Zoomed-in stats (0 <= threshold <= 1)
##################################
thresholds_zoom = [t for t in thresholds if 0 <= t <= 1]
zoom = {}
if thresholds_zoom:
    arr = np.array(thresholds_zoom)
    zoom['count'] = int(len(arr))
    zoom['mean'] = safe_float(np.mean(arr))
    zoom['median'] = safe_float(np.median(arr))
    zoom['std_dev'] = safe_float(np.std(arr, ddof=1)) if len(arr) > 1 else None
    zoom['sem'] = safe_float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None
    zoom['min'] = safe_float(np.min(arr))
    zoom['max'] = safe_float(np.max(arr))
    zoom['25th_percentile'] = safe_float(np.percentile(arr, 25))
    zoom['75th_percentile'] = safe_float(np.percentile(arr, 75))

    if len(arr) > 1:
        t_score = scipy.stats.t.ppf(0.975, df=len(arr) - 1)
        sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
        ci_range = t_score * sem
        ci_lower = np.mean(arr) - ci_range
        ci_upper = np.mean(arr) + ci_range
        zoom['95%_CI_lower'] = safe_float(ci_lower)
        zoom['95%_CI_upper'] = safe_float(ci_upper)
    else:
        zoom['95%_CI_lower'] = None
        zoom['95%_CI_upper'] = None

##################################
## Inject stats into YAML-ready data structure
##################################

data['averages'] = stats
data['averages_0to1'] = zoom
data['stats_summary'] = {
    'total_entries': int(total_entries),
    'missing_thresholds': int(missing_entries),
    'valid_thresholds': int(len(thresholds))
}

##################################
## Write back to the original YAML file
##################################

with open(input_path, 'w') as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(f"Stats added to: {input_path}")

