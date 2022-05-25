from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

class Data_split(torch.nn.Module):
      def __init__(self,num_samples):
        super(Data_split, self).__init__()
        self.num_samples = num_samples
        self.x = 0
        self.y = 0
        self.adj = 0
        self.num_classes = 0
        self.idx_train= list()
        self.idx_val = list()
        self.idx_test = list()