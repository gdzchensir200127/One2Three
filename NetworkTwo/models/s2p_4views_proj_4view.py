import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy

def conv2d_block(in_c, out_c):
	return nn.Sequential(
		nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
		nn.BatchNorm2d(out_c),
		nn.ReLU(),
	)

def linear_block(in_c, out_c):
	return nn.Sequential(
		nn.Linear(in_c, out_c),
		nn.BatchNorm1d(out_c),
		nn.ReLU(),
	)

def pixel_bias(outViewN, outW, outH, renderDepth):
	# X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
	# X, Y = X.float(), Y.float() # [H,W]
	initTile = torch.cat([
		# X.repeat([outViewN, 1, 1]), # [V,H,W]
		# Y.repeat([outViewN, 1, 1]), # [V,H,W]
		torch.ones([outViewN, outH, outW]).float() * renderDepth, 
		torch.zeros([outViewN, outH, outW]).float(),
	], dim=0) # [4V,H,W]

	return initTile.unsqueeze_(dim=0) # [1,4V,H,W]



class ConvModule(nn.Module):
	'''
	Encoder same as the S2PModel encoder
	'''
	def __init__(self,options = None,in_ch=9,out_ch=32,):
		super(ConvModule,self).__init__()
		self.options = options
		self.in_ch= in_ch
		self.out_ch= out_ch
		self.actvn = nn.ReLU()
	
		self.conv3 = nn.Conv1d(self.in_ch,self.out_ch,3,padding=1)
		self.conv5 = nn.Conv1d(self.in_ch,self.out_ch,5,padding=2)
		self.conv7 = nn.Conv1d(self.in_ch,self.out_ch,7,padding=3)
		self.conv9 = nn.Conv1d(self.in_ch,self.out_ch,9,padding=4)

	def forward(self,sig):
		out = self.actvn(self.conv3(sig)+self.conv5(sig)+self.conv7(sig)+self.conv9(sig))
		return out
  
class PoolModule(nn.Module):
	'''
	Encoder same as the S2PModel encoder
	'''
	def __init__(self,options = None,poolFunc='Max', kernel_size = 2, pad=0):
		super(PoolModule,self).__init__()
		self.options = options
		if poolFunc=='Max':
			self.pool = nn.MaxPool1d(kernel_size=kernel_size, padding=pad)
		else:
			self.pool = nn.AvgPool1d(kernel_size=kernel_size, padding=pad)

	def forward(self,x):
		out = self.pool(x)
		return out
  
class LinearModule(nn.Module):
	'''
	Encoder same as the S2PModel encoder
	'''
	def __init__(self,in_ch=640,out_ch=320,):
		super(LinearModule,self).__init__()
	
		self.LinearM = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.BatchNorm1d(out_ch),
			nn.ReLU(),
		)

	def forward(self,x):
		out = self.LinearM(x)
		return out

class ComplexEncoder(nn.Module):
	'''
	Encoder same as the S2PModel encoder
	'''
	def __init__(self,options = None,):
		self.options = options
		super(ComplexEncoder,self).__init__()
		self.actvn = nn.ReLU()
	
		# Conv module
		self.conv1Real = ConvModule(in_ch=9,out_ch=32)
		self.conv1Imag = ConvModule(in_ch=9,out_ch=32)
	
		self.conv2 = ConvModule(in_ch=32,out_ch=64)
		self.conv3 = ConvModule(in_ch=64,out_ch=128)
		self.conv4 = ConvModule(in_ch=128,out_ch=256)
		self.conv5 = ConvModule(in_ch=256,out_ch=512)


		# Pooling
		self.pool1 = PoolModule(poolFunc='Max',kernel_size=4)
		self.pool2 = PoolModule(poolFunc='Max',kernel_size=4)
		self.pool3 = PoolModule(poolFunc='Max',kernel_size=4, pad=1)
		self.pool4 = PoolModule(poolFunc='Max',kernel_size=2)
		self.pool5 = PoolModule(poolFunc='Max',kernel_size=2)

		# FC
		self.fc_1 = LinearModule(5632,1024)
		self.fc_2 = LinearModule(1024,512)
		self.fc_3 = nn.Linear(512,128)
	

	def forward(self,real,imag):

		#Layer 1 : [Batch,Sample,Antenna]->[8,9,2800] -> [8,32,700]
		out_1_real = self.conv1Real(real)
		out_1_imag = self.conv1Imag(imag)
		out_1 = out_1_real+out_1_imag
		out_1 = self.pool1(out_1)

		#Layer 2 : -> [8,64,175]
		out_2 = self.conv2(out_1)
		out_2 = self.pool2(out_2)

		#Layer 3 : -> [8,128,44]
		out_3 = self.conv3(out_2)
		out_3 = self.pool3(out_3)

		#Layer 4 : -> [8,256,22]
		out_4 = self.conv4(out_3)
		out_4 = self.pool4(out_4)

		#Layer 5 : -> [8,512,11]
		out_5 = self.conv5(out_4)
		out_5 = self.pool5(out_5)
		
		out = out_5.view([-1, 5632])
		out = self.fc_1(out)
		out = self.fc_2(out)
		out = self.fc_3(out)

		return out

def deconv2d_block(in_c, out_c):
	return nn.Sequential(
		nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
		nn.BatchNorm2d(out_c),
		nn.ReLU(),
	)
def pixel_bias(outViewN, outW, outH, renderDepth):
	# X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
	# X, Y = X.float(), Y.float() # [H,W]
	initTile = torch.cat([
		# X.repeat([outViewN, 1, 1]), # [V,H,W]
		# Y.repeat([outViewN, 1, 1]), # [V,H,W]
		torch.ones([outViewN, outH, outW]).float() * renderDepth, 
		torch.zeros([outViewN, outH, outW]).float(),
	], dim=0) # [4V,H,W]

	return initTile.unsqueeze_(dim=0) # [1,4V,H,W]
class Decoder(nn.Module):
	"""Build Decoder"""
	def __init__(self, outViewN, outW, outH, renderDepth):
		super(Decoder, self).__init__()
		self.outViewN = outViewN

		self.relu = nn.ReLU()
		self.fc1 = linear_block(512, 1024)
		self.fc2 = linear_block(1024, 2048)
		self.fc3 = linear_block(2048, 4096)
		self.deconv1 = deconv2d_block(256, 192)
		self.deconv2 = deconv2d_block(192, 128)
		self.deconv3 = deconv2d_block(128, 96)
		self.deconv4 = deconv2d_block(96, 64)
		self.deconv5 = deconv2d_block(64, 48)
		self.pixel_conv = nn.Conv2d(48, outViewN*2, 1, stride=1, bias=False)
		self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

	def forward(self, x):
		x = self.relu(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		x = x.view([-1, 256, 4, 4])
		x = self.deconv1(F.interpolate(x, scale_factor=2))
		x = self.deconv2(F.interpolate(x, scale_factor=2))
		x = self.deconv3(F.interpolate(x, scale_factor=2))
		x = self.deconv4(F.interpolate(x, scale_factor=2))
		x = self.deconv5(F.interpolate(x, scale_factor=2))
		x = self.pixel_conv(x) + self.pixel_bias.to(x.device)
		XYZ, maskLogit = torch.split(
			x, [self.outViewN, self.outViewN], dim=1)

		return XYZ, maskLogit
  
class Signal2Pixel_4views_Model(nn.Module):
	'''
	Signal2PC Network that use the signal received from two views.
	
	The network has two encoder to extract feature from I signal and Q signal, then fusion the two part of feature
	by tensor add, then put the feature to the Decoder, Decoder output the predicted point cloud, shape is [B * 1024 * 3]
	
	'''
	def __init__(self, options = None):
		super(Signal2Pixel_4views_Model,self).__init__()
		self.options = options
		if options.encoder_shared:
			self.encoder = ComplexEncoder(options=options)
		else:
			self.encoder0 = ComplexEncoder(options=options)
			self.encoder1 = ComplexEncoder(options=options)
			self.encoder2 = ComplexEncoder(options=options)
			self.encoder3 = ComplexEncoder(options=options)
		self.decoder = Decoder(outViewN=options.outViewN, outW=128, outH=128, renderDepth=options.renderDepth)
   
	def forward(self,signals):
		'''
		signals is a list of 4 views signal 
		'''
		outs = []
		if self.options.encoder_shared:
			out0 = self.encoder(signals[0]['signal_real'],signals[0]['signal_imag'])
			outs.append(out0)
			out1 = self.encoder(signals[1]['signal_real'],signals[1]['signal_imag'])
			outs.append(out1)
			out2 = self.encoder(signals[2]['signal_real'],signals[2]['signal_imag'])
			outs.append(out2)
			out3 = self.encoder(signals[3]['signal_real'],signals[3]['signal_imag'])
			outs.append(out3)
			outs = torch.concat(outs, dim = 1)
		else:
			out0 = self.encoder0(signals[0]['signal_real'],signals[0]['signal_imag'])
			outs.append(out0)
			out1 = self.encoder1(signals[1]['signal_real'],signals[1]['signal_imag'])
			outs.append(out1)
			out2 = self.encoder2(signals[2]['signal_real'],signals[2]['signal_imag'])
			outs.append(out2)
			out3 = self.encoder3(signals[3]['signal_real'],signals[3]['signal_imag'])
			outs.append(out3)
			outs = torch.concat(outs, dim = 1)
		XYZ, maskLogit = self.decoder(outs)

		return XYZ, maskLogit

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# encoder = ConvEncoder()
	# decoder = Decoder()
	options = edict()
	model = Signal2Pixel_4views_Model(options).to(device)
	signal_real = torch.randn(8, 9, 2800).to(device)
	signal_imag = torch.randn(8, 9, 2800).to(device)
	signals = {'signal_real':signal_real,'signal_imag':signal_imag}
	XYZ, maskLogit = model(signals)
	print(XYZ.shape, maskLogit.shape)