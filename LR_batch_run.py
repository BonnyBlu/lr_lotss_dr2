#!/usr/bin/env python
# coding: utf-8

'''
Version 1.0.0.0
Authors: Bonny Barkus


This is a wrap around python script to run all of the scripts and files associated
with the likelihood ratio code as a batch across HEALPix areas.

The scripts and files needed to run this are:
    - LoTSS_params.py
    - apply_lr.py
    - <input_filename.env/.yml>

This script takes the inputs:
    - working directory: the base directory that the scripts will be run from and the
      outputs will be going to. This will contain a /data folder.

'''

# Imports and setup

debug = True

import os
import sys
import glob
import yaml
from dotenv import load_dotenv, find_dotenv

dir = sys.argv[1]                                                             # Working directory to change to and run the code from and store the outputs in /data to
os.chdir(dir)                                                                 # Move to working/data directory (should be bound to container) and contain /data

try:
    BASEPATH = os.path.dirname(os.path.realpath(__file__))                    # Setting up the base path based on the working directory  
    ROOTPATH = os.path.join(BASEPATH, "..", "..")                             
except NameError as e:
    if os.path.exists("data"):
        BASEPATH = os.path.realpath(".")
        ROOTPATH = BASEPATH
    else:
        BASEPATH = os.getcwd()
        ROOTPATH = os.path.join(BASEPATH, "..", "..")

data_path = os.path.join(ROOTPATH, "data")                                    # Create the path to the /data folder
src_path = os.path.join(ROOTPATH, "src")                                      #
config_path = os.path.join(ROOTPATH, "config")                                #
out_path = os.path.join(data_path, "outputs_test")                                 # Create the path to the /data/outputs folder (for the sake of setting this up I have used outputs_test - this will need changing back to outputs)
error_file = os.path.join(out_path, "LR_error_file.yml")                      # Create the path to the file for storing the HP areas that fail to run the LR

if debug ==True:
    print(data_path, src_path, config_path, out_path, error_file)

# Find all the directories in the HP outputs folder 

dirs = glob.glob(os.path.join(out_path,'*/'))

for d in dirs:
    if debug == True:
        print(d[1+len(out_path):-1])                                          # This now pulls out the HP_directory name to use as an input in to the scripts
