#!/bin/bash

## A bash script to set off running both scripts of the LR code through singularity ##

####################
## Batching code
####################

## This batch set up runs one job on one node using 2 CPUs but not exclusively 
## so other jobs can run on this node. To change the number of CPUs used 
## change ntasks; to change/allocate memory then change mem. The output 
## files are named by default to <jobname>_<jobID> but the user can change this.

#SBATCH --job-name=LRSingularity        ##  Job Name
#SBATCH --nodes=1                       ##  Number of nodes to run tasks over
#SBATCH --ntasks=2                      ##  Requests 2 CPUs on node
#SBATCH --cpus-per-task=2               ##  Number of CPUs per task
##SBATCH --exclusive                    ##  Allocated nodes not shared with other jobs # commented out so not applied
##SBATCH --mem-40g                      ##  Memory per node # Commented out so any memory allowed

#SBATCH --error=logs/%x_%j.err          ##  Error file
#SBATCH --output=logs/%x_%j.out         ##  Log file

####################


####################
## Paths and Arguments ##
####################

WORKING_DIR=$1			                ## The path to the output directory must contain /data folder		
REGION=$2		                        ## The inputted region name will be of the form HP_###
SINGULARITY_PATH=$3                     ## The path to the singularity container
#BIND_PATHS=$4                          ## The path(s) for binding the container to
LOG_FILE=lr_errors.yml 	                ## The place where failed likelihood ratios record is stored

####################


####################
## Functions ##
####################

Help()
{
    ## Prints out the help and info below.

    cat << EOF
    
    This script combines both the scripts needed to run the LR together into one
    single executable. It takes a single input file which contains the location 
    of the radio, Gaussian and optical catalogues, and outputs an error file and
    the LR fits files into the same location:

    - Arg 1:    This needs to be inputted by the user and is the directory which
                the code will work in and needs to contain the /data folder which 
                holds the input data and is where the outputs will be stored.
    - Arg 2:    This will be automatically inputted when batched or can be manually
                inputted when dealing with single region. It is the region (or HP_###) 
                that is inputted into the scripts this will come from the name of 
                the input folders and is the main variable by which the scripts 
                search and save the data under.
    - Arg 3:    This is the pathway to the singularity container that will run the LR.
    - Arg 4:    This is currently set to lr_errors.yml and is used to store the 
                errors from apply_lr.py. Will need to be added as an arguement if you
                want to change the name.

                                ** NOTES ** 
   
    1.  The scripts will look for /data and /config in the working directory. It will 
        be placing the intermediatory outputs in the /data folder, and will be looking 
        for the two setup files in the /config folder.
        
    2.  The outputs directory should be in place from the previous use of the 
        HEALPix code. There should be separate folders for each ofthe HEALPix regions, 
        named HP_###. It is these folders (and naming conventions) that will be used to 
        insert the region to the scripts, and direct to the correct output files.
    
    3.  The user has enter in the input file, whether they wish for the radio catalogues 
        to be used; whether they want the nearest neighbour catalogue created; and if they 
        are after the initial parameters to be calculated or, just the final LRs.

    The options for this function are:

    -h:         Prints this help and info.
    
EOF
}

Setup()
{
    '''
    This setup function will setup the conda and terminal environments and pathways.
    This will need to be run in the singularity container before the two scripts are
    run.
    *** Check if this is needed ***
    If the user needs to set anything that is not already set in the container, then 
    this function is in plce for the user to create their own configuration set up.
    '''
    
}

####################


####################
## Options ##
####################

while getopts ":sh" option; do

    case ${option} in
    
        s) # Sets up the environment and PYTHONPATH
            Setup
            exit 0
            ;;
        
        h) # Pulls up the help and info
            Help
            exit 0
            ;;
        
        \?) # Invaild input
            echo "Invalid option see -h (help) for help and information."
            exit 0
            ;;
    esac
done
        
####################


####################
## Main code ##
####################

## This code will run both the scripts required for the  likelihood ratio one after the other
## using the input file, an intermediate save file, and the output file.

# This section will call the inputs from the .yaml file, this will tell the script whether it is dealing with gaussians/nearest
# neighbours, and if both the first and second scripts need running.

eval "$(${WORKING_DIR}/config/call_yaml.py ${WORKING_DIR}/config/inputs.yml lr_inputs gaussian nearest thres_calc apply_calc)"

echo "Calculating the Gaussians: $GAUSSIAN"
echo "Calculating the nearest neighbours: $NEAREST"
echo "Calculating the thresholds: $THRES_CALC"
echo "Calculating the final LRs: $APPLY_CALC"
echo "Thresholds are stored in: $PARAMS_NAME"

# Compute region suffix (e.g., '099' from 'hp_099')
REGION_SUFFIX="${REGION:3}"

# Set PARAMS_OUTPUT_DIR based on GAUSSIAN
if [ "$GAUSSIAN" = "True" ]; then
    PARAMS_OUTPUT_DIR="/data/lr_outputs/idata/${PARAMS_NAME}gauss_${REGION_SUFFIX}"
else
    PARAMS_OUTPUT_DIR="/data/lr_outputs/idata/${PARAMS_NAME}${REGION_SUFFIX}"
fi


if [ "$THRES_CALC" = "True" ]; then
    echo "The thresholds and parameters are being calculated for ${REGION}. This will overwrite the previous details."

    singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/notebooks/lr_tests/LoTSS_params.py ${WORKING_DIR} ${REGION}

    if [ "$APPLY_CALC" = "True" ]; then
        echo "Continuing with the likelihood ratio calculation."

        singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/scripts/lr/apply_lr.py ${WORKING_DIR} ${REGION}
    fi

elif [ "$APPLY_CALC" = "True" ]; then
    if [ ! -d "$PARAMS_OUTPUT_DIR" ]; then
        echo "Threshold parameters directory (${PARAMS_OUTPUT_DIR}) not found. Calculating thresholds first."

        singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/notebooks/lr_tests/LoTSS_params.py ${WORKING_DIR} ${REGION}
    fi

    echo "Continuing with the likelihood ratio calculation."

    singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/scripts/lr/apply_lr.py ${WORKING_DIR} ${REGION}

else
    echo "Neither threshold nor likelihood ratio calculation is requested. Exiting."
fi

exit































