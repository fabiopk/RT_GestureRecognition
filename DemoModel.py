import torch
import torch.nn as nn
import math

class FullModel(nn.Module):
	
	def __init__(self, batch_size, seq_lenght=8):
		super(FullModel, self).__init__()
		
		class CNN2D(nn.Module):
			def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8, in_channels=3):
				super(CNN2D, self).__init__()
				self.conv1 = self._create_conv_layer(in_channels=in_channels, out_channels=16)
				self.conv2 = self._create_conv_layer(in_channels=16, out_channels=32)
				self.conv3 = self._create_conv_layer_pool(in_channels=32, out_channels=64)
				self.conv4 = self._create_conv_layer_pool(in_channels=64, out_channels=128)
				self.conv5 = self._create_conv_layer_pool(in_channels=128, out_channels=256)
				cnn_output_shape = int(256*(image_size/(2**4))**2)

			def forward(self, x):
				batch_size, frames, channels, width, height = x.shape
				x = x.view(-1, channels, width, height)
				x = self.conv1(x)
				x = self.conv2(x)
				x = self.conv3(x)
				x = self.conv4(x)
				x = self.conv5(x)
				return x
			
			def _create_conv_layer(self,in_channels, out_channels, kernel_size=(3,3), padding=(1,1)):
				return nn.Sequential(
						nn.Conv2d(in_channels,out_channels, kernel_size, padding=padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(),
					)

			def _create_conv_layer_pool(self,in_channels, out_channels, kernel_size=(3,3), padding=(1,1), pool=(2,2)):
				return nn.Sequential(
						nn.Conv2d(in_channels,out_channels, kernel_size, padding=padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(),
						nn.MaxPool2d(pool)
					)

		class CNN3D(nn.Module):
			def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8):
				super(CNN3D, self).__init__()
				self.conv1 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(1,1,1))
				self.conv2 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2,2,2))
				self.conv3 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2,1,1))
				self.conv4 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2,2,2))

			def forward(self, x):
				batch_size, channels, frames, width, height = x.shape
				x = self.conv1(x)
				x = self.conv2(x)
				x = self.conv3(x)
				x = self.conv4(x)
				return x

			def _create_conv_layer(self,in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)):
				return nn.Sequential(
						nn.Conv3d(in_channels,out_channels, kernel_size, padding=padding),
						nn.BatchNorm3d(out_channels),
						nn.ReLU(),
					)

			def _create_conv_layer_pool(self,in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1), pool=(1,2,2)):
				return nn.Sequential(
						nn.Conv3d(in_channels,out_channels, kernel_size, padding=padding),
						nn.BatchNorm3d(out_channels),
						nn.ReLU(),
						nn.MaxPool3d(pool)
					)

			
		class Combiner(nn.Module):
			
			def __init__(self, in_features):
				super(Combiner, self).__init__()
				self.linear1 = self._create_linear_layer(in_features , in_features//2)
				self.linear2 = self._create_linear_layer(in_features//2 , 1024)
				self.linear3 = self._create_linear_layer(1024 , 27)
			
			def forward(self, x):
				x = self.linear1(x)
				x = self.linear2(x)
				x = self.linear3(x)
				return x;
	
			def _create_linear_layer(self, in_features, out_features, p=0.6):
				return nn.Sequential(
					nn.Linear(in_features, out_features),
					nn.Dropout(p=p)
				)

		self.rgb2d = CNN2D(batch_size)
		self.rgb3d = CNN3D(batch_size)
		self.combiner = Combiner(4608)

		self.batch_size = batch_size
		self.seq_lenght = seq_lenght
		self.steps = 0
		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf

	def forward(self, x):
		self.batch_size = x.shape[0]
		x = self.rgb2d(x)
		batch_and_frames, channels, dim1, dim2 = x.shape
		x = x.view(self.batch_size, -1, channels, dim1, dim2).permute(0,2,1,3,4)
		x = self.rgb3d(x)
		x = x.view(self.batch_size, -1)
		x = self.combiner(x)

		if self.training:
			self.steps += 1

		return x
