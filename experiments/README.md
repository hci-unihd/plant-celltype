# Experiments reproducibility guide
Before running the scripts:
* Install and activate the `pct` environment
* Install `palntcelltype` (or create a soft link for `plantcelltype` here).
* All scripts inside the `experiments` directory are set up to be run from inside the `expertiments`.

## Reproduce results for the baseline
In order to reproduce the baselines scores one need to:
* Downloads `baseline_checkpoints.zip` from [here](https://heibox.uni-heidelberg.de/published/celltypegraph-benchmark/)
and extract it inside the `experiments` directory. (zip file check `md5sum: 82ea2e1cf3d0e9eee342938b7c583174`).
* Run the `experiments/reproduce_baseline.py` script.

## Grid search
* Run the `experiments/node_grid_search.py` script.
* Downloads `notebooks` from [here](https://heibox.uni-heidelberg.de/published/celltypegraph-benchmark/)

## Additional experiments for reproducing additional results and plots in the paper
* Run the `features_importance_type.py`, `grs_importance.py`, `depth.py` scripts.
* Downloads `notebooks` from [here](https://heibox.uni-heidelberg.de/published/celltypegraph-benchmark/)
* Check name to summarize the results and produce the plots.
