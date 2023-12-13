# Run a baseline model in BasicTS framework.


import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(3) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="models/STPE_D2STGNN_2/PreST_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="models/D2STGNN/METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="models/PreST_WaveNet/TSFormer_PEMS04.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="models/PreST_WaveNet/STFormer_PEMS04.py", help="training config")
    parser.add_argument("-c", "--cfg", default="models/STPE_D2STGNN_4/PreST_d2stgnn_PEMS04.py", help="training config")
    parser.add_argument("-g", "--gpus", default="2", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
