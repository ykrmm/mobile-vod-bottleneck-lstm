from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Tuple
from utils import box_utils
from collections import namedtuple
from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
	"""
	return nn.Sequential(
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(in_channels), kernel_size=kernel_size,
			   groups=int(in_channels), stride=stride, padding=padding),
		nn.ReLU(),
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=1),
	)

def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(int(inp), int(oup), 3, stride, 1, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU(inplace=True)
			)
def conv_dw(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(int(inp), int(inp), 3, stride, 1, groups=int(inp), bias=False),
				nn.BatchNorm2d(int(inp)),
				nn.ReLU(inplace=True),

				nn.Conv2d(int(inp), int(oup), 1, 1, 0, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU(inplace=True),
			)
class MatchPrior(object):
	def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
		if type(gt_boxes) is np.ndarray:
			gt_boxes = torch.from_numpy(gt_boxes)
		if type(gt_labels) is np.ndarray:
			gt_labels = torch.from_numpy(gt_labels)
		boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
												self.corner_form_priors, self.iou_threshold)
		boxes = box_utils.corner_form_to_center_form(boxes)
		locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
		return locations, labels

class BottleneckLSTMCell(nn.Module):
	def __init__(self, input_channels, hidden_channels):
		super(BottleneckLSTMCell, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		self.num_features = 4
		self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3, groups=self.input_channels, stride=1, padding=1)
		self.Wy  = nn.Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
		self.Wi  = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, groups=self.hidden_channels, bias=False)  
		self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)

		self.Wci = None
		self.Wcf = None
		self.Wco = None
		logging.info("Initializing weights of lstm")
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x, h, c):
		x = self.W(x)
		y = torch.cat((x, h),1) #concatenate input and hidden layers
		i = self.Wy(y) #reduce to hidden layer size
		b = self.Wi(i)	#depth wise 3*3
		ci = torch.sigmoid(self.Wbi(b) + c * self.Wci)
		cf = torch.sigmoid(self.Wbf(b) + c * self.Wcf)
		cc = cf * c + ci * torch.relu(self.Wbc(b))
		co = torch.sigmoid(self.Wbo(b) + cc * self.Wco)
		ch = co * torch.relu(cc)
		return ch, cc

	def init_hidden(self, batch_size, hidden, shape):
		if self.Wci is None:
			self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
			self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
			self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
		else:
			assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
			assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
		return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
				Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
				)

class BottleneckLSTM(nn.Module):
	def __init__(self, input_channels, hidden_channels, height, width, batch_size):
		super(BottleneckLSTM, self).__init__()
		self.input_channels = int(input_channels)
		self.hidden_channels = int(hidden_channels)
		self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels)
		(h, c) = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, shape=(height, width))
		self.hidden_state = h
		self.cell_state = c

	def forward(self, input):
		new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
		self.hidden_state = new_h
		self.cell_state = new_c
		return self.hidden_state

def crop_like(x, target):
	if x.size()[2:] == target.size()[2:]:
		return x
	else:
		height = target.size()[2]
		width = target.size()[3]
		crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
		crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
	# fixed indexing for PyTorch 0.4
	return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


class MobileNetV1(nn.Module):
	def __init__(self, num_classes=1024, alpha=1):
		super(MobileNetV1, self).__init__()
		# upto conv 12
		self.model = nn.Sequential(
			conv_bn(3, 32*alpha, 2),
			conv_dw(32*alpha, 64*alpha, 1),
			conv_dw(64*alpha, 128*alpha, 2),
			conv_dw(128*alpha, 128*alpha, 1),
			conv_dw(128*alpha, 256*alpha, 2),
			conv_dw(256*alpha, 256*alpha, 1),
			conv_dw(256*alpha, 512*alpha, 2),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			)
		logging.info("Initializing weights of base net")
		self._initialize_weights()
		#self.fc = nn.Linear(1024, num_classes)
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x):
		x = self.model(x)
		return x


class SSD(nn.Module):
	def __init__(self,num_classes, batch_size, alpha = 1, is_test=False, config = None):
		super(SSD, self).__init__()
		# Decoder
		self.is_test = is_test
		self.config = config
		self.num_classes = num_classes
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)
		self.conv13 = conv_dw(512*alpha, 1024*alpha, 2)
		self.bottleneck_lstm1 = BottleneckLSTM(input_channels=1024*alpha, hidden_channels=256*alpha, height=10, width=10, batch_size=batch_size)
		self.fmaps_1 = nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.bottleneck_lstm2 = BottleneckLSTM(input_channels=256*alpha, hidden_channels=64*alpha, height=5, width=5, batch_size=batch_size)
		self.fmaps_2 = nn.Sequential(	
			nn.Conv2d(in_channels=int(64*alpha), out_channels=int(32*alpha), kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=32*alpha, out_channels=64*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.bottleneck_lstm3 = BottleneckLSTM(input_channels=64*alpha, hidden_channels=16*alpha, height=3, width=3, batch_size=batch_size)
		self.fmaps_3 = nn.Sequential(	
			nn.Conv2d(in_channels=int(16*alpha), out_channels=int(8*alpha), kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=8*alpha, out_channels=16*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.lstm4 = BottleneckLSTM(input_channels=16*alpha, hidden_channels=16*alpha, height=2, width=2, batch_size=batch_size)
		self.fmaps_4 = nn.Sequential(	
			nn.Conv2d(in_channels=int(16*alpha), out_channels=int(8*alpha), kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=8*alpha, out_channels=16*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.lstm5 = BottleneckLSTM(input_channels=16*alpha, hidden_channels=16*alpha, height=1, width=1, batch_size=batch_size)
		self.regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=64*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=16*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=16*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(16*alpha), out_channels=6 * 4, kernel_size=1),
		])

		self.classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=64*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=16*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=16*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(16*alpha), out_channels=6 * num_classes, kernel_size=1),
		])

		logging.info("Initializing weights of ssd")
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def compute_header(self, i, x):
		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.view(confidence.size(0), -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.view(location.size(0), -1, 4)

		return confidence, location

	def forward(self, x):
		confidences = []
		locations = []
		header_index=0
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.conv13(x)
		x = self.bottleneck_lstm1(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_1(x)
		x=self.bottleneck_lstm2(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_2(x)
		x=self.bottleneck_lstm3(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_3(x)
		x = self.lstm4(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_4(x)
		x = self.lstm5(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		confidences = torch.cat(confidences, 1)
		locations = torch.cat(locations, 1)
		
		if self.is_test:
			confidences = F.softmax(confidences, dim=2)
			boxes = box_utils.convert_locations_to_boxes(
				locations, self.priors, self.config.center_variance, self.config.size_variance
			)
			boxes = box_utils.center_form_to_corner_form(boxes)
			return confidences, boxes
		else:
			return confidences, locations

class MobileVOD(nn.Module):
	def __init__(self, pred_enc, pred_dec):
		super(MobileVOD, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		

	def forward(self, seq):
		x = self.pred_encoder(seq)
		confidences, locations = self.pred_decoder(x)
		return confidences , locations

	def detach_hidden(self):
		self.pred_decoder.bottleneck_lstm1.hidden_state.detach_()
		self.pred_decoder.bottleneck_lstm1.cell_state.detach_()
		self.pred_decoder.bottleneck_lstm2.hidden_state.detach_()
		self.pred_decoder.bottleneck_lstm2.cell_state.detach_()
		self.pred_decoder.bottleneck_lstm3.hidden_state.detach_()
		self.pred_decoder.bottleneck_lstm3.cell_state.detach_()
		self.pred_decoder.lstm4.hidden_state.detach_()
		self.pred_decoder.lstm4.cell_state.detach_()
		self.pred_decoder.lstm5.hidden_state.detach_()
		self.pred_decoder.lstm5.cell_state.detach_()
		
		


