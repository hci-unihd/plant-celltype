# plant-celltype
This repository contains the code for all experiments in the submitted manuscript. The dataset download will be handle autmonatically by the plant-celltype-graph-benchmark.

# Requirements
- linux
- Anaconda/miniconda

# dependencies
- python >= 3.7
- h5py
- pyaml
- pytorch
- pytorch_geometric
- torchmetrics

## Install dependencies using conda
- for cuda 11.1
```
conda create -n plant-ct -c rusty1s -c pytorch -c nvidia -c conda-forge -c cpape numpy scipy matplotlib scikit-image h5py pyaml jupyterlab tqdm scikit-spatial elf nifty pytorch torchvision cudatoolkit=11.1 pytorch-lightning pytorch-geometric
```
- for cuda 10.2
```
conda create -n plant-ct -c rusty1s -c pytorch -c conda-forge -c cpape numpy scipy matplotlib scikit-image h5py pyaml jupyterlab tqdm scikit-spatial elf nifty pytorch torchvision cudatoolkit=10.2 pytorch-lightning pytorch-geometric
```
- for cpu only 
```
conda create -n plant-ct -c rusty1s -c pytorch -c conda-forge -c cpape numpy scipy matplotlib scikit-image h5py pyaml jupyterlab tqdm scikit-spatial elf nifty pytorch torchvision cpuonly pytorch-lightning pytorch-geometric napari plotly python=3.9 
```

## Install pctg-benchmark
```
conda activate plant-ct
cd [path-to]/plant-celltype-graph-benchmark
pip install .
```

## optional dependencies for visualization
```
pip install 'napari[pyqt5]'
pip install plotly==5.0.0
```

## run experiments
All experiments reported in the manuscript are self-contained in `experiments`, please check the README.md inside the experiment directory for 
additional instructions.


## features
Features at all different reference systems can be runned by 
