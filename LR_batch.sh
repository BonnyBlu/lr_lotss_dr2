#!/bin/bash

## A bash script to set off the batch running of the LR code through singularity ##

####################
## Paths and Arguments
####################

_DIR=$1			## The path to the output directory		
SINGULARITY_PATH=$2		## The path to the Singulairty container - Check name
DATA_PATH=$3			## The path to the mosaics - unlikely to change
			## The path to the location of the scripts
LOG_FILE= 	## The place where failed mosaics record is stored

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
    - Arg 4:    Is the path to the location of the scripts, particularly the 
                catalogue concat script.
    - Arg 5:    Is the name of the log file. Currently set to failed_mosaics.log.

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

Clear()
{

    ## This removes the symlinks from each folder.

    for d in "${MOSAIC_PATH}"*/ ; do 
    
        MOSAIC_DIR="${PYBDSF_DIR}$(basename ${d})/"
        #echo "${d}mosaic-blanked.fits" "${MOSAIC_DIR}mosaic-blanked.fits"
        #echo "${SINGULARITY_PATH}" "${MOSAIC_DIR}ddf-tmp.sif"
    
        echo "Unlinking symlink from ${MOSAIC_DIR}"
        #unlink "${MOSAIC_DIR}ddf-tmp.sif"              ## symlink to container - check name - If they hae been used and need to be removed
        unlink "${MOSAIC_DIR}mosaic-blanked.fits"      ## Symlink to mosaic - check name
     
    done

}

####################


####################
## Options
####################

while getopts ":ch" option; do

    case ${option} in
    
        c) # Clears the symlinks
            Clear
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
## Main code
####################

## This sections checks if the code has already been run; clears a folder of the same name
## and starts again, if it was mid run. Here the code makes the folders and the symlinks.

job_ids=()                                              ##  Set up an empty array called job_ids to store the output job_ids

for d in "${MOSAIC_PATH}"*/ ; do                        ##  Change back to MOSAIC_PATH
    
    MOSAIC_DIR="${PYBDSF_DIR}$(basename ${d})/"
    #echo "${d}mosaic-blanked.fits" "${MOSAIC_DIR}mosaic-blanked.fits"

    if [ -d "${MOSAIC_DIR}" ] && [ -f "${MOSAIC_DIR}mosaic-blanked--final.srl.fits" ]; then     ## Look for folder and final file

        echo "PyBDSF already completed on ${MOSAIC_DIR}."                                       ## If there state it is complete
        
    elif [ -d "${MOSAIC_DIR}" ] && [ ! -f "${MOSAIC_DIR}mosaic-blanked--final.srl.fits" ]; then     ## Look for folder and not final file

        echo "Clearing ${MOSAIC_DIR} and creating new folder with symlinks"         ## State if this is the case and will clear it
    
        rm -r "${MOSAIC_DIR}"           ## Clear folder and all contents
    
        mkdir "${MOSAIC_DIR}"           ## Remake folder and symlinks - continue with batch
        
        #ln -s "${SINGULARITY_PATH}" "${MOSAIC_DIR}ddf-tmp.sif"              ## Symlink to container - check name - If you want to use
        ln -s "${d}mosaic-blanked.fits" "${MOSAIC_DIR}mosaic-blanked.fits"  ## Symlink to mosaic - check name
   
        job_id=$(sbatch PyBDSF_Singularity.sh "${MOSAIC_DIR}" "${MOSAIC_PATH}" "${SINGULARITY_PATH}" | awk '{print $4}') ##  This batches PyBDSF_Singularity and stores the job_id in the array
        
        echo "Running ${MOSAIC_DIR} with the following batch number: ${job_id}"    ##  This echos the directory the job is being run on

        job_ids+=("${job_id}")

    elif [ ! -d "${MOSAIC_DIR}" ]; then                 ## Check if folder does not exist

        echo "Creating ${MOSAIC_DIR} with symlinks"     ## Stating will make folder and symlinks

        mkdir "${MOSAIC_DIR}"                           ## Creates folder - continues with batch
    
        #ln -s "${SINGULARITY_PATH}" "${MOSAIC_DIR}ddf-tmp.sif"              ## Symlink to container - check name - if you want to use
        ln -s "${d}mosaic-blanked.fits" "${MOSAIC_DIR}mosaic-blanked.fits"  ## Symlink to mosaic - check name    
   
        job_id=$(sbatch PyBDSF_Singularity.sh "${MOSAIC_DIR}" "${MOSAIC_PATH}" "${SINGULARITY_PATH}" | awk '{print $4}') ##  This batches PyBDSF_Singularity and stores the job_id in the array
     
        echo "Running ${MOSAIC_DIR} with the following batch number: ${job_id}"    ##  This echos the directory the job is being run on
        
        job_ids+=("${job_id}")
        
        #echo "${job_ids}"
        
    fi     
     
done


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
    sleep 300                                           ##  Check every 5 minutes
    echo "Finished Sleep"

done


## This section runs the un-batching code after the batch jobs have run

echo "Un-batching the PyBDSF output into ConcatCats folder"

#echo "${SCRIPT_PATH}ConcatCats.sh"

sbatch "${SCRIPT_PATH}ConcatCats.sh" "${PYBDSF_DIR}" "${MOSAIC_PATH}" "${SINGULARITY_PATH}" "${SCRIPT_PATH}"

####################
