import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L


class SelfIRModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(SelfIRModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = ['UNET_MSE', 'UNET_REG', 'UNET_Blur', 'Total']
		self.visual_names = ['data_blur', 'data_gt_noise', 'data_out', 'data_gt']
		self.model_names = ['UNET'] 
		self.optimizer_names = ['UNET_optimizer_%s' % opt.optimizer]

		unet = TUNET(opt)
		self.netUNET = N.init_net(unet, opt.init_type, opt.init_gain, opt.gpu_ids)

		self.noise_adder = N.AugmentNoise(style=opt.noisetype)

		if self.isTrain:		
			self.optimizer_UNET = optim.Adam(self.netUNET.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
		
			self.optimizers = [self.optimizer_UNET]

			self.criterionMSE = N.init_net(L.MSELoss(), gpu_ids=opt.gpu_ids)
			self.criterionMseMask = N.init_net(L.MseMaskLoss(), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.data_blur = input['blur_img'].to(self.device)
		self.data_gt = input['gt_img'].to(self.device)
		self.data_gt_noise = input['gt_noise'].to(self.device)
		self.image_paths = input['fname']

	def forward(self):
		if self.isTrain:
			self.data_gt_noise = self.noise_adder.add_train_noise(self.data_gt_noise)

			mask1, mask2 = N.generate_mask_pair(self.data_gt_noise)
			noisy_sub1 = N.generate_subimages(self.data_gt_noise, mask1)
			self.blur_sub1 = N.generate_subimages(self.data_blur, mask1)
			self.noisy_sub2 = N.generate_subimages(self.data_gt_noise, mask2)

			self.data_out = self.netUNET(noisy_sub1, self.blur_sub1)

			with torch.no_grad():
				self.data_mask = N.blur_mask_gen(self.data_out.detach(), self.blur_sub1, \
					 kernel=self.opt.crpp_patch, t=0.99)  

				full_out = self.netUNET(self.data_gt_noise, self.data_blur)
				self.full_out_sub1 = N.generate_subimages(full_out, mask1)
				self.full_out_sub2 = N.generate_subimages(full_out, mask2)
		else:
			self.data_out = self.netUNET(self.data_gt_noise, self.data_blur)

	def backward(self, epoch):
		self.loss_UNET_MSE = self.criterionMSE(self.data_out, self.noisy_sub2).mean()
		self.loss_UNET_REG = self.criterionMSE(self.data_out - self.noisy_sub2, 
		                                       self.full_out_sub1 - self.full_out_sub2).mean() * 2
		
		self.loss_UNET_Blur = self.opt.blur_loss * self.criterionMseMask(
			                  self.data_out, self.blur_sub1, self.data_mask).mean()

		self.loss_Total = self.loss_UNET_MSE + self.loss_UNET_REG + self.loss_UNET_Blur
		self.loss_Total.backward()

	def optimize_parameters(self, epoch):
		self.forward()
		self.optimizer_UNET.zero_grad()
		self.backward(epoch)
		self.optimizer_UNET.step()


class TUNET(nn.Module):
	def __init__(self, opt):
		super(TUNET, self).__init__()
		self.opt = opt
		in_channel = 3

		self.mean_head = N.MeanShift(sign=-1)
		self.mean_tail = N.MeanShift(sign=1)

		self.EB0 = N.EncodeBlock(in_channel, 48, flag=False)
		self.EB1 = N.EncodeBlock(48, 48, flag=True)
		self.EB2 = N.EncodeBlock(48, 48, flag=True)
		self.EB3 = N.EncodeBlock(48, 48, flag=True)
		self.EB4 = N.EncodeBlock(48, 48, flag=True)
		self.EB5 = N.EncodeBlock(48, 48, flag=True)
		self.EB6 = N.EncodeBlock(48, 48, flag=False)

		self.EB0_2 = N.EncodeBlock(in_channel, 48, flag=False)
		self.EB1_2 = N.EncodeBlock(48, 48, flag=True)
		self.EB2_2 = N.EncodeBlock(48, 48, flag=True)
		self.EB3_2 = N.EncodeBlock(48, 48, flag=True)
		self.EB4_2 = N.EncodeBlock(48, 48, flag=True)
		self.EB5_2 = N.EncodeBlock(48, 48, flag=True)
		self.EB6_2 = N.EncodeBlock(48, 48, flag=False)

		self.DB1 = N.DecodeBlock(48*4, 96, 48)
		self.DB2 = N.DecodeBlock(144, 96, 48)
		self.DB3 = N.DecodeBlock(144, 96, 48)
		self.DB4 = N.DecodeBlock(144, 96, 96)
		self.DB5 = N.DecodeBlock(96+in_channel*2, 64, 32, in_channel, flag=True)  
		
		self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
		
		self.concat_dim = 1
		
	def forward(self, noise, blur):
		blur = self.mean_head(blur)
		noise = self.mean_head(noise)

		out_EB0 = self.EB0(blur)
		out_EB1 = self.EB1(out_EB0)
		out_EB2 = self.EB2(out_EB1)
		out_EB3 = self.EB3(out_EB2)
		out_EB4 = self.EB4(out_EB3)
		out_EB5 = self.EB5(out_EB4)
		out_EB6 = self.EB6(out_EB5)
		
		out_EB0_2 = self.EB0_2(noise)
		out_EB1_2 = self.EB1_2(out_EB0_2)
		out_EB2_2 = self.EB2_2(out_EB1_2)
		out_EB3_2 = self.EB3_2(out_EB2_2)
		out_EB4_2 = self.EB4_2(out_EB3_2)
		out_EB5_2 = self.EB5_2(out_EB4_2)
		out_EB6_2 = self.EB6_2(out_EB5_2)

		out_EB6_up = F.interpolate(out_EB6, out_EB4.shape[2:], mode='bilinear', align_corners=True)
		out_EB6_up_2 = F.interpolate(out_EB6_2, out_EB4.shape[2:], mode='bilinear', align_corners=True)
		in_DB1 = torch.cat((out_EB6_up, out_EB6_up_2, out_EB4, out_EB4_2), self.concat_dim)
		out_DB1 = self.DB1((in_DB1))
		
		out_DB1_up = self.Upsample(out_DB1)
		in_DB2 = torch.cat((out_DB1_up, out_EB3, out_EB3_2), self.concat_dim)
		out_DB2 = self.DB2((in_DB2))
		
		out_DB2_up = self.Upsample(out_DB2)
		in_DB3 = torch.cat((out_DB2_up, out_EB2, out_EB2_2), self.concat_dim)
		out_DB3 = self.DB3((in_DB3))
		
		out_DB3_up = self.Upsample(out_DB3)
		in_DB4 = torch.cat((out_DB3_up, out_EB1, out_EB1_2), self.concat_dim)
		out_DB4 = self.DB4((in_DB4))
		
		out_DB4_up = self.Upsample(out_DB4)
		in_DB5 = torch.cat((out_DB4_up, blur, noise), self.concat_dim)
		out_DB5 = self.DB5(in_DB5)
		
		out_DB5 = self.mean_tail(out_DB5)
		return out_DB5

