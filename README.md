# Analysis of Pseudo-Labeling for Online Source-Free Universal Domain Adaptation

This is the official repository to the paper "Analysis of Pseudo-Labeling for Online Source-Free Universal Domain Adaptation".

## Usage
### Preparation
- Clone this repository
- Install the requirements by running `pip install -r requirements.txt`
- Download datasets into the folder [data](data).

### Perform the simulation
To perform the domain adaptation with simulated pseudo-labeling, edit the corresponding config file [target_adaptation.yaml](configs/target_adaptation.yaml) to select the desired scenario and run the following command: `python main.py fit --config configs/target_adaptation.yaml`
