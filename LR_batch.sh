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
    
    This script runs the batching script for the Likelihood Ratio (LR) 
    Singularity container on a Slurm cluster. It is used in conjunction with 
    PLR_run_scripts.sh. Ideally all the scripts must be in the same folder to 
    run, or the pathway to LR__run_scripts.sh must be changed in the sbatch 
    code line. The arguments to the script are as 
    follows:

    - Arg 1:    Is the working directory the user wishes to work from. This 
                must contain both the /data folder and the /config folder.
    - Arg 2:    The pathway to the directory containing the healpix outputs.
    - Arg 3:    The pathway to the singularity container.


                            ** NOTE ** 
    1.  The user must have a /logs/ folder in the directory you run this script in
    for the log files of the batch jobs to save to.
    
    2.  The working directory must contain the folders /data and /config (containing
        the two setup files)
    
    3.  Please check all the pathways that are inputted.

    The options for this function are:

    -h:         Prints this help and info. 
    
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

        ## Checking to see if it has an empty status
        if [[ -z "${status}" ]]; then
            echo "Job ${job_id} has no status information (likely completed or purged from sacct). Assuming completed." >> "${LOG_FILE}"
            continue
        fi

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
