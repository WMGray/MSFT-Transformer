import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# from MBT_new_train import train
from evaluate import eval
feature = "ko,species"

for disease in ["EW-T2D", "C-T2D", 'Obesity', "LC", "IBD"]:
    for seed in [392,412,432,452,472]:
        eval(model="MTMFTransformer", disease=disease, feature=feature, seed=seed)