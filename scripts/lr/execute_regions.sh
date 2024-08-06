#!/bin/bash
cd ../../notebooks/lr_tests
export REGION="s0a"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="s0b"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="s13"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb

