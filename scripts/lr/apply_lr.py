# %% 
debug = True
log_out = True

from glob import glob
import multiprocessing
import pickle
import os
import sys
import yaml
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, join, MaskedColumn
from astropy import units as u
from dotenv import load_dotenv, find_dotenv

dir = sys.argv[1]                               # Working directory to change to and run the code from
os.chdir(dir)                                   # Move to working/data directory (should be bound to container)

if debug == True:
    print(os.getcwd())

REGION = sys.argv[2]
#envfile = region+'.env'

try:
    BASEPATH = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(BASEPATH, "..", "..", "data")
except NameError:
    if os.path.exists("data"):
        BASEPATH = "."
        data_path = os.path.join(BASEPATH, "data")
    else:
        BASEPATH = os.getcwd()
        data_path = os.path.join(BASEPATH, "..", "..", "data")

ROOTPATH = os.path.join(BASEPATH, "..", "..")
config_path = os.path.join(dir, "config")
#lr_path = os.path.join(data_path, "lr_outputs")
idp = os.path.join(data_path, "lr_outputs", "idata")
hp_path = os.path.join(data_path, "outputs", REGION)
#log_file = os.path.join(data_path, "outputs", "lr_outputs.yml")

# Using .yml input file
with open(os.path.join(config_path, "inputs.yml"), "r") as ymlfile:
    cfg_all = yaml.safe_load(ymlfile)

lr_inputs = cfg_all.get("lr_inputs", {})

gauss = lr_inputs['gaussian']
nearest = lr_inputs['nearest']
#hp_path = os.path.join(out_path, REGION)
#print(gauss)
#print(nearest)

# This calls the column names in from the input file so that they are not hard coded throughout the rest of the script

VIS_col = lr_inputs['VIS_col']
NIR_col1 = lr_inputs['NIR_col1']
NIR_col2 = lr_inputs['NIR_col2']
RA_rad = lr_inputs['RA_rad']
DEC_rad = lr_inputs['DEC_rad']
RA_opt = lr_inputs['RA_opt']
DEC_opt = lr_inputs['DEC_opt']
PA_rad = lr_inputs['PA_rad']
Maj_rad = lr_inputs['Maj_rad']
E_Maj_rad = lr_inputs['E_Maj_rad']
E_Min_rad = lr_inputs['E_Min_rad']


# Using .yml input file
print('gaussian set to', gauss)

# This changes the name of the outputs file so that the thresholds can be obtained for each lr calculation
# This can be commented out if different log files to collect the thresholds are not needed
suffix = ""

if gauss:
    suffix += "_gauss"
else:
    suffix += "_radio"

if lr_inputs.get("nearest", False):
    suffix += "_nn"

# Construct the log file name
#job_id = os.environ.get("SLURM_JOB_ID", "local")  # fallback to "local" if running outside SLURM
job_id = 'eu_001'

log_file = os.path.join(data_path, "outputs", "tmp", f"lr_outputs_{job_id}{suffix}.yml")

thres_file = os.path.join(data_path, "outputs", "average_thresholds.yml")

#os.makedirs(os.path.dirname(log_file), exist_ok=True) # This makes sure that the tmp directory exists

if debug == True:
    print(BASEPATH)
    print(data_path)
    print(config_path)
    print(idp)
    print(hp_path)

sys.path.append(os.path.join(BASEPATH, '..', '..', 'src'))
from mltier1 import MultiMLEstimator, parallel_process, get_sigma_all

'''
Bonny's addition of logging the Error's and Outputs
'''

# Function to initialize the log file
def initialize_log_file():
    with open(log_file, "w") as f:
        yaml.safe_dump({}, f)  # Start with an empty dictionary

def log_outputs(region, error, details=None, threshold=None):
    try:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Load existing log data if file exists
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_data = yaml.safe_load(f) or {}
        else:
            log_data = {}

        if region not in log_data:
            log_data[region] = {"logs": []}

        log_entry = {"error": error, "details": details if details else ""}
        if threshold is not None and isinstance(threshold, np.floating):
            log_entry["threshold"] = float(threshold)

        log_data[region]["logs"].append(log_entry)

        with open(log_file, "w") as f:
            yaml.safe_dump(log_data, f, default_flow_style=False)

    except Exception as e:
        print(f"Error logging output: {e}")

'''
End of additions, be sure to check code to remove/comment out
the appropriate code lines.

'''

# Configuration

# Old version before using the .yml file for the inputs
#load_dotenv(find_dotenv(envfile))
#COMBINED_DATA_PATH = data_path + os.getenv("COMBINED_DATA_PATH")
#PARAMS_PATH = lr_path + os.getenv("PARAMS_PATH")
#THRESHOLD = os.getenv("THRESHOLD")
#RADIO_CATALOGUE = data_path + os.getenv("RADIO_CATALOGUE")
#OUTPUT_RADIO_CATALOGUE = data_path + os.getenv("OUTPUT_CATALOGUE")

# Default config parameters
#base_optical_catalogue = COMBINED_DATA_PATH
#params = pickle.load(open(PARAMS_PATH, "rb"))
#colour_limits = np.array([0.7, 1.2, 1.5, 2. , 2.4, 2.8, 3.1, 3.6, 4.1])
#threshold = float(THRESHOLD)
#max_major = 15
#radius = 15


if (gauss, nearest) == (True, True):
    print('Calculating the Likelihood Ratio for the nearest neighbours of the healpix regions of the Gaussian catalogue.')
    print(f'Currently calculating region {REGION}')
    PARAMS_PATH = os.path.join(idp, lr_inputs["params_name"]+'gauss_'+str(REGION[3:]), 'lofar_params_'+str(REGION)+'.pckl')
    RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["gauss_in"]+'nn_'+str(REGION[3:])+'.fits')
    OUTPUT_RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["gauss_in"]+'nn_lr_'+str(REGION[3:])+'.fits')
elif (gauss, nearest) == (True, False):
    print('Calculating the Likelihood Ratio for the healpix regions of the Gaussian catalogue.')
    print(f'Currently calculating region {REGION}')
    PARAMS_PATH = os.path.join(idp, lr_inputs["params_name"]+'gauss_'+str(REGION[3:]), 'lofar_params_'+str(REGION)+'.pckl')
    RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["gauss_in"]+str(REGION[3:])+'.fits')
    OUTPUT_RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["gauss_in"]+'lr_'+str(REGION[3:])+'.fits')
elif (gauss, nearest) == (False, True):
    print('Calculating the Likelihood Ratio for the nearest neighbours of the healpix regions of the radio catalogue.')
    print(f'Currently calculating region {REGION}')
    PARAMS_PATH = os.path.join(idp, lr_inputs["params_name"]+str(REGION[3:]), 'lofar_params_'+str(REGION)+'.pckl')
    RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["rad_in"]+'nn_'+str(REGION[3:])+'.fits')
    OUTPUT_RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["rad_in"]+'nn_lr_'+str(REGION[3:])+'.fits')
else:
    print('Calculating the Likelihood Ratio for the healpix regions of the radio catalogue.')
    print(f'Currently calculating region {REGION}')
    PARAMS_PATH = os.path.join(idp, lr_inputs["params_name"]+str(REGION[3:]), 'lofar_params_'+str(REGION)+'.pckl')
    RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["rad_in"]+str(REGION[3:])+'.fits')
    OUTPUT_RADIO_CATALOGUE = os.path.join(hp_path, lr_inputs["rad_in"]+'lr_'+str(REGION[3:])+'.fits')

with open(thres_file, "r") as f:
    thresholds = yaml.safe_load(f)

if suffix not in thresholds:
    raise SystemExit(f"Threshold section for '{suffix}' not found in average_thresholds.yml.")

try:
    THRESHOLD = thresholds[suffix]["averages_0to1"]["median"]
except Exception as e:
    if log_out:
        log_outputs(REGION, "ThresholdError", details=f"Could not retrieve global median from 'averages_0to1': {e}")
    raise SystemExit(f"Threshold could not be retrieved from global 'averages_0to1': {e}. Will need to run previous steps.")



# Default config parameters
COMBINED_DATA_PATH = os.path.join(hp_path, lr_inputs["opt_nn_in"]+str(REGION[3:])+'.fits')
base_optical_catalogue = COMBINED_DATA_PATH
params = pickle.load(open(PARAMS_PATH, "rb"))
threshold = float(THRESHOLD)
colour_limits = np.array(lr_inputs["colour_limits_post"])
max_major = lr_inputs["max_major"]
radius = lr_inputs['radius']

if threshold == 0:
    if log_out == True:
        log_outputs(REGION, "ThresholdError", details = "The threshold has a value of Zero and the LRs have not been calculated.")
        raise SystemExit("Threshold is equal to Zero and the LR calculation has been exited")


#print('PARAMS_PATH', PARAMS_PATH)
#print('RADIO_CATALOGUE', RADIO_CATALOGUE)
#print('OUTPUT_RADIO_CATALOGUE', OUTPUT_RADIO_CATALOGUE)
#print('base_optical_catalogue', base_optical_catalogue)
#print('threshold', threshold)
#print('max_major', max_major)
#print('colour_limits_post', colour_limits_post)
                  
#sys.exit('Testing the imports from the .yml file')



# input_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.fits"))
# output_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.lr.fits"))
input_catalogue = RADIO_CATALOGUE
output_catalogue = OUTPUT_RADIO_CATALOGUE

# %% 
bin_list, centers, Q_0_colour, n_m, q_m = params

## Load the catalogues
print("Load optical catalogue")
combined = Table.read(base_optical_catalogue)
print("Load input catalogue")
lofar = Table.read(input_catalogue)

## Get the coordinates
coords_combined = SkyCoord(combined[RA_opt], 
                        combined[DEC_opt], 
                        unit=(u.deg, u.deg), 
                        frame='icrs')
coords_lofar = SkyCoord(lofar[RA_rad], 
                    lofar[DEC_rad], 
                    unit=(u.deg, u.deg), 
                    frame='icrs')

## Get the colours for the combined catalogue
print("Get auxiliary columns")
combined["colour"] = combined[VIS_col] - combined[NIR_col1]
combined_aux_index = np.arange(len(combined))
combined_legacy = (
    ~np.isnan(combined[VIS_col]) & 
    ~np.isnan(combined[NIR_col1]) & 
    ~np.isnan(combined[NIR_col2])
)
combined_wise =(
    np.isnan(combined[VIS_col]) & 
    ~np.isnan(combined[NIR_col1])
)
combined_wise2 =(
    np.isnan(combined[VIS_col]) & 
    np.isnan(combined[NIR_col1]) &
    ~np.isnan(combined[NIR_col2])
)

# Start with the W2-only, W1-only, and "less than lower colour" bins
colour_bin_def = [{"name":"only W2", "condition": combined_wise2},
                {"name":"only WISE", "condition": combined_wise},
                {"name":"-inf to {}".format(colour_limits[0]), 
                "condition": (combined["colour"] < colour_limits[0])}]

# Get the colour bins
for i in range(len(colour_limits)-1):
    name = "{} to {}".format(colour_limits[i], colour_limits[i+1])
    condition = ((combined["colour"] >= colour_limits[i]) & 
                (combined["colour"] < colour_limits[i+1]))
    colour_bin_def.append({"name":name, "condition":condition})

# Add the "more than higher colour" bin
colour_bin_def.append({"name":"{} to inf".format(colour_limits[-1]), 
                    "condition": (combined["colour"] >= colour_limits[-1])})

# Apply the categories
combined["category"] = =1 # changed from np.nan to cover for unmatched entries and will trigger a 0 probability
for i in range(len(colour_bin_def)):
    combined["category"][colour_bin_def[i]["condition"]] = i


## Define number of CPUs
n_cpus_total = multiprocessing.cpu_count()
n_cpus = max(1, n_cpus_total-1)
print(f"Use {n_cpus} CPUs")

## Start matching
print("X-match")
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined, radius*u.arcsec
    )
idx_lofar_unique = np.unique(idx_lofar)
def apply_ml(i, likelihood_ratio_function):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    
    category = combined["category"][idx_0].astype(int)
    mag = combined[VIS_col][idx_0]
    mag[category == 0] = combined[NIR_col2][idx_0][category == 0]
    mag[category == 1] = combined[NIR_col1][idx_0][category == 1]
    
    lofar_ra = lofar[i][RA_rad]
    lofar_dec = lofar[i][DEC_rad]
    lofar_pa = lofar[i][PA_rad]
    lofar_maj_err = lofar[i][E_Maj_rad]
    lofar_min_err = lofar[i][E_Min_rad]
    c_ra = combined[RA_opt][idx_0]
    c_dec = combined[DEC_opt][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                    lofar_ra, lofar_dec, 
                    c_ra, c_dec, c_ra_err, c_dec_err)

    lr_0 = likelihood_ratio_function(mag, d2d_0.arcsec, sigma_0_0, det_sigma, category)
    
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[idx_0[chosen_index]], # Index
            (d2d_0.arcsec)[chosen_index],                        # distance
            lr_0[chosen_index]]                                  # LR
    return result
likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)
def ml(i):
    return apply_ml(i, likelihood_ratio)
print("Run LR")
res = parallel_process(idx_lofar_unique, ml, n_jobs=1)
lofar["lr"] = np.nan                   # Likelihood ratio
lofar["lr_dist"] = np.nan              # Distance to the selected source
lofar["lr_index"] = np.nan             # Index of the optical source in combined
(lofar["lr_index"][idx_lofar_unique], 
    lofar["lr_dist"][idx_lofar_unique], 
    lofar["lr"][idx_lofar_unique]) = list(map(list, zip(*res)))

## 
lofar["lrt"] = lofar["lr"]
lofar["lrt"][np.isnan(lofar["lr"])] = 0
lofar["lr_index_sel"] = lofar["lr_index"]
lofar["lr_index_sel"][lofar["lrt"] < threshold] = np.nan

## Save combined matches
combined["lr_index_sel"] = combined_aux_index.astype(float)
print("Combine catalogues")
pwl = join(lofar, combined, join_type='left', keys='lr_index_sel')
print("Clean catalogues")
# Iterate over columns in the table
for col in pwl.colnames:
    if not isinstance(pwl[col], MaskedColumn):
        
        # Check if there are NaN values in the column (assuming the column is numeric)
        if np.issubdtype(pwl[col].dtype, np.number):  # Only check numeric columns for NaNs
            nan_mask = np.isnan(pwl[col])  # Find where NaNs are
            if np.any(nan_mask):  # If there are any NaN values
                print(f"Column {col} has NaN values and will be converted to a MaskedColumn.")
                
                # Convert the column to a MaskedColumn with NaNs masked
                pwl[col] = MaskedColumn(pwl[col], mask=nan_mask)
        else:
            print(f"Column {col} is not numeric and cannot contain NaNs.")

    # Now we are sure the column is either originally a MaskedColumn or has been converted to one.
    if isinstance(pwl[col], MaskedColumn):
        fv = pwl[col].fill_value

        # Check and update the fill_value
        if (isinstance(fv, np.float64) and (fv != 1e+20)):
            print(f"Updating fill_value for column {col}. Old fill_value: {fv}")
            pwl[col].fill_value = 1e+20
        
print("Save output")
pwl[RA_opt].name = "ra"
pwl[DEC_opt].name = "dec"
pwl[RA_rad].name = "RA"
pwl[DEC_rad].name = "DEC"
pwl.filled().write(output_catalogue, format="fits", overwrite=True)

    
