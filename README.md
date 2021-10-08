## Learning 3D Representations of Molecular Chirality with Invariance to Bond Rotations

![ScreenShot](/figures/InterRotoInvariance.png)

This directory contains the model architectures and experimental setups used for ChIRo, SchNet, DimeNet++, and SphereNet on the four tasks considered in the preprint: 

**Learning 3D Representations of Molecular Chirality with Invariance to Bond Rotations**

These four tasks are:

1) Contrastive learning to cluster conformers of different stereoisomers in a learned latent space
2) Classification of chiral centers as R/S
3) Classification of the sign (+/-; l/d) of rotated circularly polarized light
4) Ranking enantiomers by their docking scores in an enantiosensitive protein pocket.

The exact data splits used for tasks (1), (2), and (4) can be downloaded from:

https://figshare.com/s/e23be65a884ce7fc8543

See the appendix of "Learning 3D Representations of Molecular Chirality with Invariance to Bond Rotations" for details on how the datasets for task (3) were extracted and filtered from the commercial Reaxys database.

------------------------------------------------------------------------------------------------------------------------------

This directory is organized as follows:

* Subdirectory ```model/``` contains the implementation of ChIRo.
    * ```model/alpha_encoder.py``` contains the network architecture of ChIRo
    * ```model/embedding_functions.py``` contains the featurization of the input conformers (RDKit mol objects) for ChIRo.
    * ```model/datasets_samplers.py``` contains the Pytorch / Pytorch Geometric data samplers used for sampling conformers in each training batch.
    * ```model/train_functions.py``` and ```model/train_models.py``` contain supporting training/inference loops for each experiment with ChIRo.
    * ```model/optimization_functions.py``` contains the loss functions used in the experiments with ChIRo.
    
    * Subdirectory ```model/gnn_3D/``` contains the implementations of SchNet, DimeNet++, and SphereNet used for each experiment.
        * ```model/gnn_3D/schnet.py``` contains the publicly available code for SchNet, with adaptations for readout.
        * ```model/gnn_3D/dimenet_pp.py``` contains the publicly available code for DimeNet++, with adaptations for readout.
        * ```model/gnn_3D/spherenet.py``` contains the publicly available code for SphereNet, with adaptations for readout.
        * ```model/gnn_3D/train_functions.py``` and ```model/gnn_3D/train_models.py``` contain the training/inference loops for each experiment with SchNet, DimeNet++, or SphereNet.
        * ```model/gnn_3D/optimization_functions.py``` contains the loss functions used in the experiments with SchNet, DimeNet++, or SphereNet.

* Subdirectory ```params_files/``` contains the hyperparameters used to define exact network initializations for ChIRo, SchNet, DimeNet++, and SphereNet for each experiment. The parameter .json files are specified with a random seed = 1, and the first fold of cross validation for the l/d classifcation task. For the experiments specified in the paper, we use random seeds = 1,2,3 when repeating experiments across three training/test trials.

* Subdirectory ```training_scripts/``` contains the python scripts to run each of the four experiments, for each of the four 3D models ChIRo, SchNet, DimeNet++, and SphereNet. Before running each experiment, move the corresponding training script to the parent directory.

* Subdirectory ```hyperopt/``` contains hyperparameter optimization scripts for ChIRo using Raytune.


* Subdirectory ```experiment_analysis/``` contains jupyter notebooks for analyzing results of each experiment.

* Subdirectory ```paper_results/``` contains the parameter files, model parameter dictionaries, and loss curves for each experiment reported in the paper.

-----------------------------------------------------------------------------------------------------------------------------

To run each experiment, first create a conda environment with the following dependencies:

* python = 3.8.6
* pytorch = 1.7.0
* torchaudio = 0.7.0
* torchvision = 0.8.1
* torch-geometric = 1.6.3
* torch-cluster = 1.5.8
* torch-scatter = 2.0.5
* torch-sparce = 0.6.8
* torch-spline-conv = 1.2.1
* numpy = 1.19.2
* pandas = 1.1.3
* rdkit = 2020.09.4
* scikit-learn = 0.23.2
* matplotlib = 3.3.3
* scipy = 1.5.2
* sympy = 1.8
* tqdm = 4.58.0

Then, download the datasets (with exact training/validation/test splits) from https://figshare.com/s/e23be65a884ce7fc8543 and place them in a new directory ```final_data_splits/```

You may then run each experiment by calling:

```console
python training_{experiment}_{model}.py params_files/params_{experiment}_{model}.json {path_to_results_directory}/
```


For instance, you can run the docking experiment for ChIRo with a random seed of 1 (editable in the params .json file) by calling:

```console
python training_binary_ranking.py params_files/params_binary_ranking_ChIRo.json results_binary_ranking_ChIRo/
```

After training, this will create a results directory containing model checkpoints, best model parameter dictionaries, and results on the test set (if applicable).
    
