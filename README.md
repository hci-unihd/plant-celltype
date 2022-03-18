# plant-celltype
This repository contains the code for all experiments in the submitted manuscript. The dataset download will be handled automatically by the plant-celltype-graph-benchmark.

# Requirements
- Linux
- Anaconda / miniconda

# Dependencies
- python >= 3.7
- h5py
- pyaml
- pytorch
- pytorch_geometric
- torchmetrics

## Install dependencies using conda
- for cuda 11.3
```
conda create -n pct -c rusty1s -c lcerrone -c pytorch -c conda-forge ctg-benchmark cudatoolkit=11.3 tifffile scikit-image scikit-spatial python-elf pytorch-lightning 
```
- for cuda 10.2
```
conda create -n pct -c rusty1s -c lcerrone -c pytorch -c conda-forge ctg-benchmark cudatoolkit=10.2 tifffile scikit-image scikit-spatial python-elf pytorch-lightning
```
- for cpu only 
```
conda create -n pct -c rusty1s -c lcerrone -c pytorch -c conda-forge ctg-benchmark cpuonly tifffile scikit-image scikit-spatial python-elf pytorch-lightning 
```

## Install plant-celltype
With the `plant-ct` environment active, executed from the root directory:
```
pip install .
```

## Optional dependencies for visualization
```
pip install 'napari[pyqt5]'
pip install plotly==5.0.0
```
## Reproduce experiments
All experiments reported in the manuscript are self-contained in [experiments](experiments), please check the README.md inside the experiment directory for 
additional instructions.

## Features
Features can be computed from segmentation by running:  
```
python run_dataprocessing.py -c example_config/build_dataset/CONFIG-NAME.yaml
```

## Predictions
To run prediction on new segmentation data using a pretrained model
* Configure the pipeline by editing the prediction
[config](example_config/node_predictions/predict_from_segmentation.yaml).
* Run the pipeline by:
```
python run_dataprocessing.py -c example_config/node_predictions/predict_from_segmentation.yaml
```