# MSFT-Transformer
MSFT-Transformer: A Multi-Stage Fusion Tabular Transformer for Disease Prediction using Metagenomic Data

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

## Note:
The rest of the code will be open sourced after the paper is accepted, so stay tuned.

## Citation
