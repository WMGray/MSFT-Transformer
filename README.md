# MTMF-Transformer
MTMF-Transformer: A Multimodal Tabular Transformer with Multi-Level Fusion for Disease Prediction using Biomedical Data

## Datasets
For detailed data preprocessing workflows, please refer to the associated publication and the CIAE repository: [CIAE](https://github.com/gwcde/CIAE).

## Quick Setup Guide:
1. Step 1: Change the current working directory to the location where you want to install MTMF-Transformer
2. Step 2: Clone the repository using git command:
```bash
git clone git@github.com:WMGray/MTMF-Transformer.git
```
3. Step 3: Create a virtual environment using conda:
```bash
conda create -n mtmf python=3.10
conda activate mtmf
```
4. Step 4: Install the required packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```
5. Step 5: Run the main.py file to train the model:
    - You can use the '-uc' parameter to reproduce the result, where the config files are stored in the 'Config' folder.
   ```python
   python main.py -d EW-T2D -uc --gpu 0
   ```
   - You can also specify parameters to train the model, for example:
   ```python
    python main.py -d EW-T2D --gpu 0 --bs 32 --lr 0.0001 -num_b 4  # MTMF-Transformer
   ```
   Different models have different parameters. For details, you can view `main.py` file.


## Citation
``` bash
@article{wang2025msft,
  title={MSFT-transformer: a multistage fusion tabular transformer for disease prediction using metagenomic data},
  author={Wang, Ning and Wu, Minghui and Gu, Wenchao and Dai, Chenglong and Shao, Zongru and Subbalakshmi, KP},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={3},
  pages={bbaf217},
  year={2025},
  publisher={Oxford University Press}
}
```