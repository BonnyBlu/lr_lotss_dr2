#!/bin/bash

## A bash script to set off the batch running of the LR code through singularity ##

####################
## Paths and Arguments
####################

WORKING_DIR=$1		        ## The path to the working directory
OUT_DIR=$2			        ## The path to the healpix directory		
SINGULARITY_PATH=$3			## The path to the singularity container
LOG_FILE=LR_batch_log.log

####################


####################
## Functions
####################

Help()
{
    ## Prints out the help and info below.

    cat << EOF
    
    This script runs the batching and un-batching script for the PyBDSF 
    Singularity container on a Slurm cluster. It is used in conjunction with 
    PyBDSF_Singularity.sh and ConcatCats.sh. Ideally all the scripts must be 
    in the same folder to run, or the pathway to PyBDSF_Singularity.sh must 
    be changed in the sbatch code line. The first five lines can be adjusted 
    to allow the pathways to be included as arguments to the scripts as follows:

    - Arg 1:    Is the pathway to where the output is to be stored.
    - Arg 2:    Is the pathway to the Singularity container.
    - Arg 3:    Is the pathway to the mosaics. Most likely to be unchaged.


                            ** NOTE ** 
    1. You must have a /logs/ folder in the directory you run this script in
    for the log files of the batch jobs to save to.
    
    2. You must create your LOG_FILE for recording the outcomes.
    
    3. Please check the name of the Singularity container and the mosaics in the
    script; locations are indicated.

    The options for this function are:

    -h:         Prints this help and info.
    -c:         Cleans the symbolic links from the output directories.  
    
EOF
}


####################


####################
## Options
####################

while getopts ":h" option; do

    case ${option} in
    
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
## Main code
####################

#Do we want to do a check for if the data is already there or will we always go with an overwrite aspect?

## This sections checks if the code has already been run; clears a folder of the same name
## and starts again, if it was mid run. Here the code makes the folders and the symlinks.

job_ids=()                                              ##  Set up an empty array called job_ids to store the output job_ids

shopt -s nullglob

dirs=(${OUT_DIR}/*/)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "No subdirectories found in ${OUT_DIR}"
    exit 1
fi

for d in "${dirs[@]}" ; do                          ##  

    REGION=$(basename "${d}")

    echo "Submitting job for directory: ${REGION}"
   
    job_id=$(sbatch LR_run_scripts.sh "${WORKING_DIR}" "${REGION}" "${SINGULARITY_PATH}"| awk '{print $4}') ##  This batches PyBDSF_Singularity and stores the job_id in the array
        
    job_ids+=("${job_id}")
      
done


####################

## This section checks to make sure all the previous batch jobs have completed

while true; do

    active_jobs=0
    
    echo "Monitoring the following job ids: ${job_ids[@]}"
    for job_id in "${job_ids[@]}"; do                               ##  For each of the job ids in the array job_ids from above
        sleep 10
        echo "Setting status"
        ## Fetch job status awk filters out lines containing COMPLETED and dashes (with any number of spaces). gsub strips leading spaces. Prints the first field, and the first line if more than one.
        status=$(sacct -j "${job_id}" --format=State --noheader | awk '!/COMPLETED|^[[:space:]]*--+[[:space:]]*$/{gsub(/^[[:space:]]+/, ""); print $1}' | head -n 1)
        #echo "${status}"
        #echo "${LOG_FILE}"
        echo "Checking status of ${job_id}: ${status}"
        case "${status}" in
           
            ## Running or pending jobs output everything is fine and kept going by setting all_done to 0.
            RUNNING|PENDING)
                echo "Job is running normally. Status is not recorded in log file."                                
                ((active_jobs++))
                ;;
                
            ## Failed/cancelled/timeout/node_fail/revoked jobs are recorded in the output file so that the .err file can be looked at.
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|REVOKED)
                echo "Job ${job_id} ended abnormally with status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                ;;
                
            ## Transitioning jobs, will continue shortly; kept going by setting all_done to 0
            CONFIGURING|COMPLETING)
                echo "Job will complete shortly. Status is: ${status}"
                ((active_jobs++))
                ;;
                
            ## Temporary status jobs; kept going by setting all_done to 0. Records to log file.
            SUSPENDED|PREEMPTED)
                echo "Job ${job_id} is in a temporary state: ${status}. Manual check might be needed. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                ((active_jobs++))
                ;;
                
            ## Deals with any other status that might occur.  Keeps the loop going by setting all_done to 0, and records to log file.
            *)
                echo "Job ${job_id} has an unexpected status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                ((active_jobs++))
                ;;
        esac
    
    done
    
    echo "Number of current active jobs is: ${active_jobs}"
    
    if [[ "${active_jobs}" -eq 0 ]]; then
        echo "About to break"
        break
    
    fi
    
    
    echo "Entering Sleep"
    sleep 10                                           ##  Check every 5 minutes
    echo "Finished Sleep"

done


####################
