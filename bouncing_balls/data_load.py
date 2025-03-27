'''import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from data_prep.offline_input_data_gen import *
from data_prep.offline_output_data_gen import *
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
import glob

class Loader(Dataset):

	def __init__(self, im_size, n_balls, n_frames, rep_times, sample_size, mask_path=None, aux_data=None):
		super(Loader, self).__init__()
		self.im_size = im_size
		self.n_balls = n_balls
		self.n_frames = n_frames
		self.rep_times = rep_times
		self.sample_size = sample_size
		self.aux_data = True if aux_data is not None else False
		self.aux_prob = 0.5

		if mask_path:
			self.mask = sio.loadmat(mask_path)
			self.mask = self.mask[sorted(self.mask.keys())[-1]]
		else:
			self.mask = None

		if self.aux_data:
			input_path_list, output_path_list = sorted(glob.glob(aux_data+'*.png')), glob.glob(aux_data+'*.mat')
			self.input_list = []

			for im in input_path_list:
				im = Image.open(im)
				self.input_list.append(transforms.ToTensor()(im).float())

			self.output_data = sio.loadmat(output_path_list[0])
			self.output_data = torch.from_numpy(self.output_data[sorted(self.output_data.keys())[0]]).unsqueeze(1).float()

	def __getitem__(self, index):

		if self.aux_data:
			if random.random() < self.aux_prob:
				idx = random.choice(np.arange(len(self.input_list)))
				inp, out = self.input_list[idx], self.output_data[idx]

			else:
				inp, out = self.gen_example()

		else:
			inp, out = self.gen_example()

		return inp, out

	def __len__(self):
		return self.sample_size

	def gen_example(self):

		out = bounce_mat(res=self.im_size, n=self.n_balls, T=self.n_frames)
		out = np.moveaxis(out, 0, -1)
		out = np.repeat(out, self.rep_times, axis=-1)
		inp = get_streaking_image(out, self.mask)

		out = torch.from_numpy(out).unsqueeze(0).float().contiguous()
		inp = torch.from_numpy(inp).unsqueeze(0).float().contiguous()

		return inp, out

class Loader_offline(Dataset):

	def __init__(self, input_file_name, output_file_name):
		super(Loader_offline, self).__init__()
		self.input_file_name = input_file_name
		self.output_file_name = output_file_name
		self.in_file = None
		self.out_file = None

		open_file = h5py.File(output_file_name, 'r')

		self.len = open_file['data'].shape[0]

		open_file.close()

	def __getitem__(self, index):

		if not self.in_file: self.in_file = h5py.File(self.input_file_name, 'r')
		in_samples = self.in_file['data'][index]
		in_samples = torch.from_numpy(in_samples).float().unsqueeze(0)

		self.in_file.close()

		if not self.out_file: self.out_file = h5py.File(self.output_file_name, 'r')
		out_samples = np.moveaxis(self.out_file['data'][index], -1, 0)
		out_samples = (torch.from_numpy(out_samples).float() - 0.5) / 0.5

		return in_samples, out_samples

	def __len__(self):
		return self.len

class Loader_gen(Dataset):

	def __init__(self, im_size, n_balls, n_frames, sample_size):
		super(Loader_gen, self).__init__()
		self.im_size = im_size
		self.n_balls = n_balls
		self.n_frames = n_frames
		self.sample_size = sample_size

	def __getitem__(self, index):

		out = bounce_mat(res=self.im_size, n=self.n_balls, T=self.n_frames)
		out = np.moveaxis(out, 0, -1)

		out = torch.from_numpy(out[:,:,np.random.randint(self.n_frames)]).squeeze().unsqueeze(0).float().contiguous()

		return out

	def __len__(self):
		return self.sample_size

class Loader_gen_offline(Dataset):

	def __init__(self, hdf5_name):
		super(Loader_gen_offline, self).__init__()
		self.hdf5_name = hdf5_name

		open_file = h5py.File(self.hdf5_name, 'r')
		self.length = len(open_file['data'])
		open_file.close()

		self.open_file = None

	def __getitem__(self, index):

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		scene = torch.from_numpy(np.moveaxis(self.open_file['data'][index], -1, 0)).float()
		idx = np.random.randint(scene.size(0))
		img = scene[idx]

		return img.unsqueeze(0)

	def __len__(self):
		return self.length

if __name__=='__main__':

	test_dataset = Loader(200, 3, 50, 2, 100, './mask.mat')

	print(test_dataset.mask)

	inp_, out_ = test_dataset.__getitem__(10)

	print(inp_.shape, out_.shape)

	test_dataset = Loader_gen(200, 3, 50, 100)

	out_ = test_dataset.__getitem__(10)

	print(out_.shape)'''

import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import glob
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms

# =================== SHEARED IMAGE LOADER ===================
class Loader(Dataset):
    def __init__(self, image_dir, sample_size, mask_path=None):
        super(Loader, self).__init__()
        self.image_dir = image_dir  # Path to sheared images
        self.sample_size = sample_size
        
        # Load all sheared images
        self.image_list = sorted(glob.glob(os.path.join(image_dir, '*.png')))  

        print(f"✅ Using sheared images from {image_dir}")

        if mask_path:
            self.mask = torch.tensor(np.load(mask_path))  # Ensure correct format
        else:
            self.mask = None

    def __getitem__(self, index):
        # Load sheared image
        img_path = self.image_list[index % len(self.image_list)]
        image = Image.open(img_path).convert('L')
        image = transforms.ToTensor()(image).float()

        # Simulate reconstructed video using a stack of sheared images
        frame_count = min(len(self.image_list), 10)  # Simulating 10-frame motion
        video_frames = [Image.open(self.image_list[i]).convert('L') 
                        for i in range(frame_count)]
        video_frames = torch.stack([transforms.ToTensor()(img).float() for img in video_frames])

        return image, video_frames  # Image + pseudo-video sequence

    def __len__(self):
        return self.sample_size


# =================== OFFLINE DATA LOADER ===================
class Loader_offline(Dataset):
    def __init__(self, input_file_name, output_file_name):
        super(Loader_offline, self).__init__()
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.in_file = None
        self.out_file = None

        # Get dataset size without loading into memory
        with h5py.File(output_file_name, 'r') as f:
            self.len = f['data'].shape[0]

    def __getitem__(self, index):
        # Lazy loading of input file
        if self.in_file is None: 
            self.in_file = h5py.File(self.input_file_name, 'r')
        in_samples = self.in_file['data'][index]
        in_samples = torch.from_numpy(in_samples).float().unsqueeze(0)

        # Lazy loading of output file
        if self.out_file is None: 
            self.out_file = h5py.File(self.output_file_name, 'r')
        out_samples = np.moveaxis(self.out_file['data'][index], -1, 0)
        out_samples = (torch.from_numpy(out_samples).float() - 0.5) / 0.5

        return in_samples, out_samples

    def __len__(self):
        return self.len


# =================== TEST SCRIPT ===================
if __name__ == '__main__':
    # Example paths (update as needed)
    image_dir = './sheared_images/'  # Folder with sheared images
    
    test_dataset = Loader(image_dir, sample_size=100, mask_path=None)
    
    inp_, out_ = test_dataset.__getitem__(10)
    print(f"Sheared Image Shape: {inp_.shape}, Reconstructed Video Shape: {out_.shape}")

