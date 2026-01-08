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
    lr_params_run.sh and apply_lr_run.sh. Ideally all the scripts must be in 
    the same folder to run, or the pathway to LR__run_scripts.sh must be changed
    in the sbatch code line. The arguments to the script are as 
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
## Variables and arguments
####################

# Build the same suffix as in the Python script

eval "$(${WORKING_DIR}/config/call_yaml.py ${WORKING_DIR}/config/inputs.yml lr_inputs gaussian nearest thres_calc apply_calc)"

echo "Calculating the Gaussians: $GAUSSIAN"
echo "Calculating the nearest neighbours: $NEAREST"
echo "Calculating the thresholds: $THRES_CALC"
echo "Calculating the final LRs: $APPLY_CALC"

SUFFIX=""
if [[ "$GAUSSIAN" == "True" ]]; then
    SUFFIX="${SUFFIX}_gauss"
else
    SUFFIX="${SUFFIX}_radio"
fi

if [[ "$NEAREST" == "True" ]]; then
    SUFFIX="${SUFFIX}_nn"
fi



####################
## Main code
####################

if [ "$THRES_CALC" = "True" ]; then


    job_ids=()                                              ##  Set up an empty array called job_ids to store the output job_ids

    shopt -s nullglob

    dirs=("${OUT_DIR}"/*/)

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No subdirectories found in ${OUT_DIR}"
        exit 1
    fi

    for d in "${dirs[@]}" ; do                           

        REGION=$(basename "${d}")

        echo "Submitting job for directory: ${REGION}"
    
        job_id=$(sbatch lr_params_run.sh "${WORKING_DIR}" "${REGION}" "${SINGULARITY_PATH}"| awk '{print $4}') ##  This batches LRSingularity and stores the job_id in the array
            
        job_ids+=("${job_id}")

        sleep 0.5                                       ## This is to prevent the job submission from throttling
        
    done


####################

    ## This section checks to make sure all the previous batch jobs have completed

    while true; do
        active_jobs=0
        echo "Monitoring the following job ids: ${job_ids[@]}"

        # Fetch all job statuses in one call
        job_id_str=$(IFS=','; echo "${job_ids[*]}")
        sacct_output=$(sacct -j "$job_id_str" --format=JobID,State --noheader)

        for job_id in "${job_ids[@]}"; do
            echo "Setting status for job ${job_id}"

            # Extract the primary status for the job from sacct output
            status=$(echo "$sacct_output" | awk -v id="$job_id" '
                $1 ~ "^"id"($|[.])" && $2 !~ /COMPLETED|^--*$/ {
                    gsub(/^[[:space:]]+/, "", $2); print $2; exit
                }'
            )

            if [[ -z "${status}" ]]; then
                echo "Job ${job_id} has no status information (likely completed or purged). Assuming completed." >> "${LOG_FILE}"
                continue
            fi

            echo "Checking status of ${job_id}: ${status}"
            case "${status}" in
                RUNNING|PENDING)
                    echo "Job is running normally. Status is not recorded in log file."                                
                    ((active_jobs++))
                    ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|REVOKED)
                    echo "Job ${job_id} ended abnormally with status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ;;
                CONFIGURING|COMPLETING)
                    echo "Job will complete shortly. Status is: ${status}"
                    ((active_jobs++))
                    ;;
                SUSPENDED|PREEMPTED)
                    echo "Job ${job_id} is in a temporary state: ${status}. Manual check might be needed. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ((active_jobs++))
                    ;;
                *)
                    echo "Job ${job_id} has an unexpected status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ((active_jobs++))
                    ;;
            esac
        done

        echo "Number of current active jobs is: ${active_jobs}"
        if [[ "${active_jobs}" -eq 0 ]]; then
            echo "All jobs completed. Proceeding to next step."
            break
        fi

        echo "Sleeping before next check..."
        sleep 2  # sleeps for 2 seconds before re-checking
    done

    ####################

    ## Merge YAML results once all jobs are complete

    if [[ -f "${WORKING_DIR}/scripts/merge_yml_results.py" ]]; then
        #python "${WORKING_DIR}/scripts/merge_yml_results.py" "${WORKING_DIR}" "${SUFFIX}"
        singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python "${WORKING_DIR}/scripts/merge_yml_results.py" "${WORKING_DIR}" "${SUFFIX}"
        echo "Merge complete. Output stored in outputs/lr_outputs${SUFFIX}.yml"
    else
        echo "Error: ${WORKING_DIR}/scripts/merge_yml_results.py not found. Exiting." >&2
        exit 1
    fi

    ####################

    ## Calculate average threshold

    if [[ -f "${WORKING_DIR}/scripts/threshold_stats.py" ]]; then
        echo "Calculating the threshold for ${SUFFIX}."
        #python "${WORKING_DIR}/scripts/threshold_stats.py" "${WORKING_DIR}/data/outputs/lr_outputs${SUFFIX}" "${SUFFIX}"
        singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python "${WORKING_DIR}/scripts/threshold_stats.py" "${WORKING_DIR}/data/outputs/lr_outputs${SUFFIX}.yml" "${SUFFIX}"
        echo "Average threshold for ${SUFFIX} calculated, ready for LR."
    else
        echo "Error: ${WORKING_DIR}/scripts/threshold_stats.py not found. Exiting." >&2
        exit 1
fi

else
    echo "Threshold calculation not requested. Exit job run and edit config.yml if calculations are required."

fi

####################

if [ "$APPLY_CALC" = "True" ]; then

    ## Checking that the averages have successfully been calculated before moving on to the LR calculation

    YAML_FILE="${WORKING_DIR}/data/outputs/average_thresholds.yml"

    if [[ -f "${YAML_FILE}" ]] && grep -q "${SUFFIX}" "${YAML_FILE}"; then
        echo "YAML file exists and contains the averages for ${SUFFIX}. Proceeding..."
    else
        echo "Threshold calculations have not been averaged. Please run script again with THRES_CALC = True set in the config.yml file. Exiting"
        exit 1
    fi

    job_ids=()                                              ##  Set up an empty array called job_ids to store the output job_ids

    shopt -s nullglob

    dirs=("${OUT_DIR}"/*/)

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No subdirectories found in ${OUT_DIR}"
        exit 1
    fi

    for d in "${dirs[@]}" ; do                          ##  

        REGION=$(basename "${d}")

        echo "Submitting job for directory: ${REGION}"
    
        job_id=$(sbatch apply_lr_run.sh "${WORKING_DIR}" "${REGION}" "${SINGULARITY_PATH}"| awk '{print $4}') ##  This batches LRSingularity and stores the job_id in the array
            
        job_ids+=("${job_id}")

        sleep 0.5                                       ## This is to prevent the job submission from throttling
        
    done


    ####################

    ## This section checks to make sure all the previous batch jobs have completed

    while true; do
        active_jobs=0
        echo "Monitoring the following job ids: ${job_ids[@]}"

        # Fetch all job statuses in one call
        job_id_str=$(IFS=','; echo "${job_ids[*]}")
        sacct_output=$(sacct -j "$job_id_str" --format=JobID,State --noheader)

        for job_id in "${job_ids[@]}"; do
            echo "Setting status for job ${job_id}"

            # Extract the primary status for the job from sacct output
            status=$(echo "$sacct_output" | awk -v id="$job_id" '
                $1 ~ "^"id"($|[.])" && $2 !~ /COMPLETED|^--*$/ {
                    gsub(/^[[:space:]]+/, "", $2); print $2; exit
                }'
            )

            if [[ -z "${status}" ]]; then
                echo "Job ${job_id} has no status information (likely completed or purged). Assuming completed." >> "${LOG_FILE}"
                continue
            fi

            echo "Checking status of ${job_id}: ${status}"

            case "${status}" in
                RUNNING|PENDING)
                    echo "Job is running normally. Status is not recorded in log file."                                
                    ((active_jobs++))
                    ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|REVOKED)
                    echo "Job ${job_id} ended abnormally with status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ;;
                CONFIGURING|COMPLETING)
                    echo "Job will complete shortly. Status is: ${status}"
                    ((active_jobs++))
                    ;;
                SUSPENDED|PREEMPTED)
                    echo "Job ${job_id} is in a temporary state: ${status}. Manual check might be needed. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ((active_jobs++))
                    ;;
                *)
                    echo "Job ${job_id} has an unexpected status: ${status}. Recorded in log file ${LOG_FILE}" >> "${LOG_FILE}"
                    ((active_jobs++))
                    ;;
            esac
        done

        echo "Number of current active jobs is: ${active_jobs}"
        if [[ "${active_jobs}" -eq 0 ]]; then
            echo "All jobs completed. Proceeding to next step."
            break
        fi

        echo "Sleeping before next check..."
        sleep 2  # sleeps for 2 seconds before re-checking
    done

else
    echo "Likelihood ratio calculation not requested. Exit job run and edit the config.yml file if these calculations are required."

fi

####################

## Merge error YAML results once all jobs are complete

echo "All threshold jobs complete. Merging results..."

#python ${WORKING_DIR}/merge_yml_results.py "${WORKING_DIR}" "_errors${SUFFIX}"
singularity exec --bind "${WORKING_DIR}","${WORKING_DIR}/data:/Documents/lr_lotss_dr2/data/" "${SINGULARITY_PATH}" python ${WORKING_DIR}/scripts/merge_yml_results.py "${WORKING_DIR}" "_errors${SUFFIX}"
echo "Merge complete. Output stored in outputs/lr_outputs_errors${SUFFIX}.yml"




