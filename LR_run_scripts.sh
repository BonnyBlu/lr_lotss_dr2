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
   
    1.  The scripts will look for /data in the working directory and are expecting
        to find the input data within this folder (as pointed to by the input file)
        and will be placing the outputs in this folder under the outputs directory.
        
    2.  The outputs directory should have in place from the previous setup of the 
        HEALPix code separate folders for each ofthe HEALPix regions, named HP_###.
        It is these folders (and naming conventions) that will be used to insert 
        the region to the scripts, and direct the correct output files.
    
    3. 

    The options for this function are:

    -s:         Runs the Setup for the environment and PYTHONPATH
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
    '''

    conda active myenv
    alias python='/opt/conda/envs/myenv/bin/python3'
    export PYTHONPATH=/azimuth/lr_lotss_dr2/notebooks/lr_tests/:$PYTHONPATH
    export PYTHONPATH=/azimuth/lr_lotss_dr2/scripts/lr/:$PYTHONPATH
    
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

eval $(./config/call_yaml.py config/inputs.yml lr_inputs gaussian nearest thres_calc apply_calc)

echo "Calculating the Gaussians: $GAUSSIAN"
echo "Calculating the nearest neighbours: $NEAREST"
echo "Calculating the thresholds: $THRES_CALC"
echo "Calculating the final LRs: $APPLY_CALC"

# If the thresholds are to be calculated: $THRES_CALC == True

if [ "$THRES_CALC" = "True" ]; then
    echo "The thresholds and parameters are being calculated for ${REGION}. This will overwrite the previous details"

    singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/notebooks/lr_tests/LoTSS_params.py ${WORKING_DIR} ${REGION}

    echo "The thresholds and parameters have been calculated for ${REGION}. Continuing with the likelihood ratio calculation."

    singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/scripts/lr/apply_lr.py ${WORKING_DIR} ${REGION}

#singularity exec --bind /project,/project/data/:/Documents/lr_lotss_dr2/data/ /project/LR/LRcontainer_new.sif python /Documents/lr_lotss_dr2/notebooks/lr_tests/LoTSS_params.py /project hp_98

else
    echo "The thresholds and parameters are not being calculated for ${REGION}. Continuing with the likelihood ratio calculation."

    singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/"  "${SINGULARITY_PATH}" python /Documents/lr_lotss_dr2/scripts/lr/apply_lr.py ${WORKING_DIR} ${REGION}  

#singularity exec --bind /project,/project/data/:/Documents/lr_lotss_dr2/data/ /project/LR/LRcontainer_new.sif python /Documents/lr_lotss_dr2/scripts/lr/apply_lr.py /project hp_98

fi

exit






























