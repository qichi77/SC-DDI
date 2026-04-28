代码已清理完毕，去除了附带的表情符号和解释性注释，保留了所有执行逻辑和环境变量设置，同时确认了模型名称的统一。

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
from RunModel import run_SC_model, ensemble_run_SC_model
from model import SCDTI

parser = argparse.ArgumentParser(
    prog='SC-DTI',
    description='SC-DTI is model in paper: "multimodal information fusion method for drug-target interaction prediction"',
    epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=[
                    "DrugBank", "Davis", "BIOSNAP","BD2D"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['SC-DTI', 'SC-DTI-B'],
                    default='SC-DTI', help='Which model to use, "SC-DTI" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=1,
                    help='Set the K-Fold number, the default is 1')
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help='cuda number, the default is 0')

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if args.model == 'SC-DTI':
    run_SC_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=SCDTI, K_Fold=args.fold, LOSS='PolyLoss', device=device)
if args.model == 'SC-DTI-B':
    ensemble_run_SC_model(SEED=args.seed, DATASET=args.dataSetName, K_Fold=args.fold, device=device)
```