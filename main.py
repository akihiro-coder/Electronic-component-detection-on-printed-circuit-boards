import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

from matplotlib import pyplot as plt
from IPython.display import clear_output
from PIL import Image
import cv2

import os, random, gc
import re, time, json
import collections
from  ast import literal_eval
import ast

from tqdm.notebook import tqdm
import joblib

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor







if __name__ == '__main__':
    print('ok')
    # (データの確認)
    
    # 前処理
    ## 画像の選定
    ## (データの可視化)
    ## データセットの作成
    # モデル定義
    # モデル学習
    # 後処理
    ## 結果の確認
    # テストデータで推論 submission書き出し
