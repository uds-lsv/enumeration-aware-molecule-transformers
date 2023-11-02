# Enumeration-aware Molecular Transformers for Representation Learning

## Overview
We introduce a suite of neural language model tools for pre-training, fine-tuning SMILES-based molecular language models. Furthermore, we also provide recipes for semi-supervised recipes for fine-tuning these languages in low-data settings using Semi-supervised learning. Lastly, we also have open-sourced mechanisms for data augmentations and generating embeddings from pre-trained molecular language models as independent packages described below.

### [1. Enumeration-aware Molecular Transformers](https://github.com/MoleculeTransformers/enumeration-aware-molecule-transformers)
Introduces contrastive learning alongside multi-task regression, and masked language modelling as pre-training objectives to inject enumeration knowledge into pre-trained language models.
#### a. Molecular Domain Adaptation (Contrastive Encoder-based)
##### i. Architecture
![smole bert drawio](https://user-images.githubusercontent.com/6007894/233776921-41667331-1ab7-413c-92f7-4e6fad512f5c.svg)
##### ii. Contrastive Learning
<img width="1418" alt="Screenshot 2023-04-22 at 11 54 23 AM" src="https://user-images.githubusercontent.com/6007894/233777069-439c18cc-77a2-4ae2-a81e-d7e94c30a6be.png">

#### b. Canonicalization Encoder-decoder (Denoising Encoder-decoder)
<img width="702" alt="Screenshot 2023-04-22 at 11 43 06 AM" src="https://user-images.githubusercontent.com/6007894/233776512-ab6cdeef-02f1-4076-9b76-b228cbf26456.png">


## Code
You can reproduce the experiment by:
### Install depedencies
`bash
pip install -r requirements.txt
`
### 1.Pre-training the molecular transfomers
The detailed steps are for pre-training with both MLM and MTR objectives are outlined in [here](./src/1_pre_training/README.md).

### 2. Domain Adaptation with Contrastive Learning and Multitask Learning
To reproduce the domain adaptation step from our work please follow the guidelines [here](./src/2_domain_adaptation/README.md).

### 3. Finetuning
Finally for finetuning the domain adapted molecular languages on downstream tasks are explained in the accompanying notebook which can be found [here](./src/3_fine_tuning/).



## Acknowledgements
Code base adapted from:
* [SimCSE](https://github.com/princeton-nlp/SimCSE)
* [Chemberta-2](https://github.com/seyonechithrananda/bert-loves-chemistry)
