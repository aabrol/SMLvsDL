# Citation - Abrol, A., Fu, Z., Salman, M. et al. Deep learning encodes robust discriminative neuroimaging representations to outperform standard machine learning. Nat Commun 12, 353 (2021). https://doi.org/10.1038/s41467-020-20655-6 
## Reproducible Example Code (reprex) https://github.com/aabrol/SMLvsDL/tree/master/reprex
Reprex DL Classification/Regression Scripts \
Slurm script - https://github.com/aabrol/SMLvsDL/tree/master/reprex/JSA_DL.sh \
Bash for python script - https://github.com/aabrol/SMLvsDL/tree/master/reprex/run_DL.sh \
Python script - https://github.com/aabrol/SMLvsDL/tree/master/reprex/run_DL.py \
Utilies - https://github.com/aabrol/SMLvsDL/tree/master/reprex/utils.py \
DL Model - https://github.com/aabrol/SMLvsDL/tree/master/reprex/models.py \
Conda Environment - https://github.com/aabrol/SMLvsDL/tree/master/reprex/AA_DL.yml
## Detailed (previous) version 
##### Package Dependencies
conda (version 4.8.3) cudatoolkit (version 10.0.13) cudnn (version 7.6.5)\
h5py (version 2.9.0) hdf5 (version 1.10.4) hypopt (version 1.0.9)\
nipy (version 0.4.1) numpy  (version 1.17.2) nibabel (version 2.5.0)\
pandas (version 0.25.1) python (version 3.7.4) pytorch (version 1.2.0)\
scikit-learn (version 0.21.3) scipy (version 1.2.0)\
slurm (version 19.05.0) torchvision (version 0.4.0)\
pytorch-lightning (version 0.10.0)
##### Custom Utilities
utils.py\
models.py
##### Generate Data Partitions
makePartitionsUKBB.py\
makePartitionsADNI.py
##### Dimension Reduction for Standard Machine Learning Methods
JSA_DR.sh\
DR.py\
JSA_DR_ADNI.sh\
DR_ADNI.py
##### Standard Machine Learning Classifiers
JSA_SML.sh\
run_SML.py 
##### Standard Machine Learning Regressors
JSA_SML_reg.sh\
run_SML_reg.py
##### Deep Learning Classifiers
JSA_DL.sh\
run_DL.sh\
run_DL.py          
##### Deep Learning Regressors
JSA_DL_reg.sh\
run_DL_reg.sh\
run_DL_reg.py  
##### Deep Learning Embeddings Visualization
tsneProjections.py
##### Deep Learning Saliency
JSA_DL_saliency.sh\
run_DL_saliency.sh\
run_DL_saliency.py
##### Comparative Analysis 
Peng et al. model/pipeline used in Schulz et al.\
JSA_run_train.sh\
run_train.sh
