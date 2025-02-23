# Analysis of Pseudo-Labeling for Online Source-Free Universal Domain Adaptation

This is the official repository to the paper "Analysis of Pseudo-Labeling for Online Source-Free Universal Domain Adaptation".

## Usage
### Preparation
- Clone this repository
- Install the requirements by running `pip install -r requirements.txt`
- Download datasets into the folder [data](data).

### Source training
We uploaded the checkpoints of our pre-trained source models into the folder [source_models](source_models). To still do the source training yourself, edit the corresponding config file [source_training.yaml](configs/source_training.yaml) accordingly and run the following command: `python main.py fit --config configs/source_training.yaml`

### Source-only testing
To test without adaptation, i.e. to get the source-only baseline results, edit the corresponding config file [source_only_testing.yaml](configs/source_only_testing.yaml) to select the desired scenario and run the following command: `python main.py test --config configs/source_only_testing.yaml`

### Perform the simulation
To perform the domain adaptation with simulated pseudo-labeling, edit the corresponding config file [target_adaptation.yaml](configs/target_adaptation.yaml) to select the desired scenario and run the following command: `python main.py fit --config configs/target_adaptation.yaml`
