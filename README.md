# plant-celltype
This repository contains the code for all experiments in the submitted manuscript. The dataset download will be handled
automatically by the [plant-celltype-graph-benchmark](https://github.com/hci-unihd/celltype-graph-benchmark).

## Requirements
- Linux
- Anaconda / miniconda

### Dependencies
- python >= 3.8
- ctg-benchmark
- tiffile
- scikit-image
- scikit-spatial
- python-elf
- pytorch-lightning

### Install dependencies using conda
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

Additional dependencies
```
pip install class_resolver       
```

### Install plantcelltype
With the `pct` environment active, executed from the root directory:
```
pip install .
```

### Optional dependencies for visualization
```
pip install 'napari[pyqt5]'
pip install plotly==5.0.0
```
## Reproduce experiments
All experiments reported in the manuscript are self-contained in [experiments](experiments), please check the 
`README.md` inside the experiment directory for additional instructions.

## Process raw data
Features can be computed from segmentation by running:  
```
python run_dataprocessing.py -c example_config/build_dataset/CONFIG-NAME.yaml
```

## Run predictions
To run prediction on new segmentation data using a pretrained model
* Configure the pipeline by editing the prediction
[config](example_config/node_predictions/predict_from_segmentation.yaml).
* Run the pipeline by:
```
python run_dataprocessing.py -c example_config/node_predictions/predict_from_segmentation.yaml
```

## Cite
@inproceedings{cerrone2022celltypegraph,
  title={CellTypeGraph: A New Geometric Computer Vision Benchmark},
  author={Cerrone, Lorenzo and Vijayan, Athul and Mody, Tejasvinee and Schneitz, Kay and Hamprecht, Fred A},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20897--20907},
  year={2022}
}
