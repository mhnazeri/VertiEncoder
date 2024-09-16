[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official PyTorch implementation of "VertiEncoder: Self-Supervised Kinodynamic Representation Learning on Vertically Challenging Terrain".

<p align="center">
  <img src="docs/VertiCoder.svg"  height="150" width="600"/>
</p>


## Installation
Main libraries:
* [PyTorch](https://www.pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

First create a virtual env for the project. 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the latest version of PyTorch from the [official site](htpps://www.pytorch.org/). Finally, run the following:
```bash
pip install -r requirements
pip install -e .
```

## Training
To run pretext training (edit [config](vertiencoder/conf/transformer.yaml) first) then run:
```bash
./run.sh train
```

To run downstream task, edit [config](vertiencoder/conf/dt.yaml) for the downstream name (options: ['bc', 'fkd', 'ikd']) and then run:
```bash
./run.sh train_dt
```