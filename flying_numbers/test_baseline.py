from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
from train_loop import TrainLoop

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance

import torchvision.transforms as transforms

def test_model(model, data_loader, n_tests, cuda_mode, enhancement):

	model.eval()
	to_pil = transforms.ToPILImage()

	with torch.no_grad():

		for i in range(n_tests):
			sample_in, sample_out = data_loader[i]
			sample_out = sample_out.transpose(0,-1).squeeze(-1)

			if cuda_mode:
				sample_in = sample_in.cuda()

			out = model.forward(sample_in)
			out = torch.sigmoid(out)
			frames_list = []

			for j in range(out.size(-1)):

				gen_frame = out[...,j].contiguous()
				frames_list.append(gen_frame[0])

			sample_rec = torch.cat(frames_list, 0)

			save_gif(sample_out, str(i+1)+'_real.gif', enhance=False)
			save_gif(sample_rec, str(i+1)+'_rec.gif', enhance=enhancement)

def save_gif(data, file_name, enhance):

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


def plot_learningcurves(history, keys):

	for i, key in enumerate(keys):
		plt.figure(i+1)
		plt.plot(history[key])
		plt.title(key)
	
	plt.show()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Generate reconstructed samples')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--enhance', action='store_true', default=False, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	### Data options
	parser.add_argument('--im-size', type=int, default=64, metavar='N', help='H and W of frames (default: 64)')
	parser.add_argument('--n-digits', type=int, default=2, metavar='N', help='Number of bouncing digits (default: 2)')
	parser.add_argument('--n-frames', type=int, default=40, metavar='N', help='Number of frames per sample (default: 40)')
	parser.add_argument('--rep-times', type=int, default=1, metavar='N', help='Number of times consecutive frames are repeated. No rep is equal to 1 (default: 1)')
	parser.add_argument('--mask-path', type=str, default='./mask.npy', metavar='Path', help='path to encoding mask')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print(args)

	data_set = Loader(im_size=args.im_size, n_objects=args.n_digits, n_frames=args.n_frames, rep_times=args.rep_times, sample_size=args.n_tests, mask_path=args.mask_path, baseline_mode=True)

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	model = models_zoo.model_baseline(n_frames=args.n_frames)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	print(model.load_state_dict(ckpt['model_state'], strict=True))

	if args.cuda:
		model = model.cuda()

	test_model(model=model, data_loader=data_set, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance)
