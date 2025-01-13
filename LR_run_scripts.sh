#!/bin/bash

## A bash script to set off running both scripts of the LR code through singularity ##

####################
## Paths and Arguments ##
####################

WORKING_DIR=$1			                ## The path to the output directory must contain /data folder		
REGION=$2		                        ## The inputted region name will be of the form HP_###
INPUT_FILE=$3			                ## The file which holds the input data needed for the pipeline
OUTPUT_FILE=lr_thresholds.yml			## The file to store the intermediate outputs
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
    - Arg 3:    This has to be manually inputted by the user and is the main file
                which contains all the input information for the pipeline. This is 
                currently a .yml file.
    - Arg 4:    This is currently set to lr_threshold.yml and stores the 
                intermediatory outputs from LoTSS_params.py to use in apply_lr.py.
    - Arg 5:    This is currently set to lr_errors.yml and is used to store the 
                errors from apply_lr.py.

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


## To run the LoTSS_params.py file

#cd /project/repos/lr_lotss_dr2/notebooks/lr_tests         # Change to the directory which contains the script

# This runs the script from within the directory given and is the pathway inside the container. Need to change this to the correct directory once the scripts are stored in the container.

## Need to add the input file once adapted the script to take an input file
## will need to give an output file, and therefore take an intermediatory output file name??

python /project/repos/lr_lotss_dr2/notebooks/lr_tests/LoTSS_params.py ${WORKING_DIR} ${REGION}           




## To run the apply_lr.py file

#cd /project/repos/lr_lotss_dr2/scripts/lr        # Change to the directory which contains the script

# This runs the script from within the directory given and is the pathway inside the container. Need to change this to the correct directory once the scripts are stored in the container.

## Need to add the input file once adapted the script to take an input file - probably need to adapt to take two input files, one for the generic inputs and one for the thresholds
## Will need to give an output file of errors, and therefore will need to give an output file name.

python /project/repos/lr_lotss_dr2/scripts/lr/LoTSS_params.py ${WORKING_DIR} ${REGION}  














