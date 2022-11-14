import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from util.util import SSIM


operation_seed_counter = 0

def get_generator():
	global operation_seed_counter
	operation_seed_counter += 1
	g_cuda_generator = torch.Generator(device="cuda")
	g_cuda_generator.manual_seed(operation_seed_counter)
	return g_cuda_generator


def blur_mask_gen(output, blur, kernel, t):
	ssim_loss = SSIM(window_size=11, size_average=False)
	N, C, H, W = output.shape
	patch_num = (H*W) // (kernel*kernel)
	output = torch.clamp(output, 0, 1)

	def rgb2gray(rgb):
		r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
		gray = 0.2989*r + 0.5870*g + 0.1140*b
		img = F.unfold(gray, kernel_size=(kernel, kernel), padding=0, stride=kernel) # [N, C*k*k, H*W]
		img = img.view(N, 1, kernel, kernel, -1).permute(0, 4, 1, 2, 3).contiguous()
		img = img.view(N*patch_num, 1, kernel, kernel).contiguous()
		return img 

	img1 = rgb2gray(output)
	img2 = rgb2gray(blur)
	ssim_value = ssim_loss(img1, img2)
	ssim_value[ssim_value<t] = 0

	var_img1 = torch.var(img1, dim=(2, 3))
	var_img2 = torch.var(img2, dim=(2, 3))

	diff_fold = var_img1 - var_img2 + 1e-5 
	diff_fold[diff_fold>0] = 0
	diff_fold[diff_fold<0] = 1
	ssim_value = ssim_value * diff_fold.squeeze(1)

	ssim_value = ssim_value.view(N, patch_num).unsqueeze(1)
	result = ssim_value.repeat(1, kernel * kernel, 1)
	mask = F.fold(result, output_size=(H,W), kernel_size=(kernel, kernel), stride=kernel)

	return mask


def space_to_depth(x, block_size):
	n, c, h, w = x.size()
	unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
	return unfolded_x.view(n, c * block_size**2, h // block_size,
						   w // block_size)


def generate_mask_pair(img):
	# prepare masks (N x C x H/2 x W/2)
	n, c, h, w = img.shape
	mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
						dtype=torch.bool,
						device=img.device)
	mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
						dtype=torch.bool,
						device=img.device)
	# prepare random mask pairs
	idx_pair = torch.tensor(
		[[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
		dtype=torch.int64,
		device=img.device)
	rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
						 dtype=torch.int64,
						 device=img.device)
	torch.randint(low=0,
				  high=8,
				  size=(n * h // 2 * w // 2, ),
				  generator=get_generator(),
				  out=rd_idx)
	rd_pair_idx = idx_pair[rd_idx]
	rd_pair_idx += torch.arange(start=0,
								end=n * h // 2 * w // 2 * 4,
								step=4,
								dtype=torch.int64,
								device=img.device).reshape(-1, 1)
	# get masks
	mask1[rd_pair_idx[:, 0]] = 1
	mask2[rd_pair_idx[:, 1]] = 1
	return mask1, mask2


def generate_subimages(img, mask):
	n, c, h, w = img.shape
	subimage = torch.zeros(n,
						   c,
						   h // 2,
						   w // 2,
						   dtype=img.dtype,
						   layout=img.layout,
						   device=img.device)
	# per channel
	for i in range(c):
		img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
		img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
		subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
			n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
	return subimage


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
										step_size=opt.lr_decay_iters,
										gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
												   mode='min',
												   factor=0.2,
												   threshold=0.01,
												   patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
												   T_max=opt.niter,
												   eta_min=0)
	else:
		return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net


class AugmentNoise(object):
	def __init__(self, style):
		if style.startswith('gauss'):
			self.params = [
				float(p) / 255.0 for p in style.replace('gauss', '').split('_')
			]
			if len(self.params) == 1:
				self.style = "gauss_fix"
			elif len(self.params) == 2:
				self.style = "gauss_range"
		elif style.startswith('poisson'):
			self.params = [
				float(p) for p in style.replace('poisson', '').split('_')
			]
			if len(self.params) == 1:
				self.style = "poisson_fix"
			elif len(self.params) == 2:
				self.style = "poisson_range"
		print('Noise Style: %s, Noise Params: %s' % (self.style, str(self.params)))

	def add_train_noise(self, x):
		shape = x.shape # [0,1]
		if self.style == "gauss_fix":
			std = self.params[0]
			std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
			noise = torch.cuda.FloatTensor(shape, device=x.device)
			torch.normal(mean=0.0,
						 std=std,
						 generator=get_generator(),
						 out=noise)
			return x + noise
		elif self.style == "gauss_range":
			min_std, max_std = self.params
			std = torch.rand(size=(shape[0], 1, 1, 1),
							 device=x.device) * (max_std - min_std) + min_std
			noise = torch.cuda.FloatTensor(shape, device=x.device)
			torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
			return x + noise
		elif self.style == "poisson_fix":
			lam = self.params[0]
			lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
			noised = torch.poisson(lam * x, generator=get_generator()) / lam
			return noised
		elif self.style == "poisson_range":
			min_lam, max_lam = self.params
			lam = torch.rand(size=(shape[0], 1, 1, 1),
							 device=x.device) * (max_lam - min_lam) + min_lam
			noised = torch.poisson(lam * x, generator=get_generator()) / lam
			return noised


class MeanShift(nn.Conv2d):
	""" is implemented via group conv """
	def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040),
				 rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1, groups=3)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.ones(3).view(3, 1, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False


class EncodeBlock(nn.Module):
	def __init__(self, in_channel, out_channel, flag):
		super(EncodeBlock, self).__init__()
		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
		self.nonlinear = nn.LeakyReLU(0.1)
		self.MaxPool = nn.MaxPool2d(2)
		self.flag = flag
		
	def forward(self, x):
		out1 = self.conv(x)
		out2 = self.nonlinear(out1)
		if self.flag:
			out = self.MaxPool(out2)
		else:
			out = out2
		return out


class DecodeBlock(nn.Module):
	def __init__(self, in_channel, mid_channel, out_channel, final_channel=3, flag = False):
		super(DecodeBlock, self).__init__()
		self.flag = flag
		self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 1)
		self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size = 3, padding = 1)

		if self.flag:
			self.conv3 = nn.Conv2d(out_channel, final_channel, kernel_size = 3, padding = 1)
		self.nonlinear1 = nn.LeakyReLU(0.1)
		self.nonlinear2 = nn.LeakyReLU(0.1)
		self.flag = flag
		
	def forward(self, x):
		out1 = self.conv1(x)
		out2 = self.nonlinear1(out1)
		
		out3 = self.conv2(out2)
		out4 = self.nonlinear2(out3)
		if self.flag:
			out = self.conv3(out4)
		else:
			out = out4
		return out