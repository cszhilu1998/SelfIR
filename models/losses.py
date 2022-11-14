# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import L1Loss, MSELoss


class MseMaskLoss(nn.Module):
	def __init__(self):
		super(MseMaskLoss, self).__init__()
		self.mse = MSELoss()
	
	def forward(self, sr, hr, mask):
		mse = torch.pow(sr - hr, 2) * mask  # b*c*h*w
		mse_sum = torch.sum(torch.sum(mse, 3), 2)  # b*c
		mask_sum = torch.sum(torch.sum(mask, 3), 2)  # b*c
		result = mse_sum / (mask_sum + 1)
		return result.mean()
