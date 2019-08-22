from torch.utils import data
import matplotlib.pyplot as plt
import cv2
import numpy as np

DEBUG = True

def four_point_transform(image, pts):

	max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

	dst = np.array([
		[0, 0],
		[image.shape[1] - 1, 0],
		[image.shape[1] - 1, image.shape[0] - 1],
		[0, image.shape[0] - 1]], dtype="float32")

	M = cv2.getPerspectiveTransform(dst, pts)
	warped = cv2.warpPerspective(image, M, (max_x, max_y))

	return warped


class DataLoader(data.Dataset):

	def __init__(self, type_):

		self.type_ = type_
		self.base_path = '<Path for Images>'
		if DEBUG:
			import os
			if not os.path.exists('cache.pkl'):
				with open('cache.pkl', 'wb') as f:
					import pickle
					from scipy.io import loadmat
					mat = loadmat('Path for gt.mat')
					pickle.dump([mat['imnames'][0][0:1000], mat['charBB'][0][0:1000], mat['txt'][0][0:1000]], f)
					print('Created the pickle file, rerun the program')
					exit(0)
			else:
				with open('cache.pkl', 'rb') as f:
					import pickle
					self.imnames, self.charBB, self.txt = pickle.load(f)
					print('Loaded DEBUG')

		else:

			from scipy.io import loadmat
			mat = loadmat('Path for gt.mat')

			total_number = mat['imnames'][0].shape[0]
			train_images = int(total_number * 0.9)

			if self.type_ == 'train':

				self.imnames = mat['imnames'][0][0:train_images]
				self.charBB = mat['charBB'][0][0:train_images]  # number of images, 2, 4, num_character

			else:

				self.imnames = mat['imnames'][0][train_images:]
				self.charBB = mat['charBB'][0][train_images:]  # number of images, 2, 4, num_character

		for no, i in enumerate(self.txt):
			all_words = []
			for j in i:
				all_words += [k for k in ' '.join(j.split('\n')).split() if k!='']
			self.txt[no] = all_words

		sigma = 10
		spread = 3
		extent = int(spread * sigma)
		self.gaussian_heatmap = np.zeros([2 * extent, 2 * extent], dtype=np.float32)

		for i in range(2 * extent):
			for j in range(2 * extent):
				self.gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
					-1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))

		self.gaussian_heatmap = (self.gaussian_heatmap / np.max(self.gaussian_heatmap) * 255).astype(np.uint8)

	def add_character(self, image, bbox):

		top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
		bbox -= top_left[None, :]
		transformed = four_point_transform(self.gaussian_heatmap.copy(), bbox.astype(np.float32))
		image[top_left[1]:top_left[1]+transformed.shape[0], top_left[0]:top_left[0]+transformed.shape[1]] += transformed
		return image

	def generate_target(self, image_size, character_bbox):

		character_bbox = character_bbox.transpose(2, 1, 0)

		channel, height, width = image_size

		target = np.zeros([height, width], dtype=np.uint8)

		for i in range(character_bbox.shape[0]):

			target = self.add_character(target, character_bbox[i])

		return target/255, np.float32(target != 0)

	def add_affinity(self, image, bbox_1, bbox_2):

		center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
		tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
		bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
		tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
		br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

		affinity = np.array([tl, tr, br, bl])

		return self.add_character(image, affinity)

	def generate_affinity(self, image_size, character_bbox, text):

		"""

		:param image_size: shape = [3, image_height, image_width]
		:param character_bbox: [2, 4, num_characters]
		:param text: [num_words]
		:return:
		"""

		character_bbox = character_bbox.transpose(2, 1, 0)

		channel, height, width = image_size

		target = np.zeros([height, width], dtype=np.uint8)

		total_letters = 0

		for word in text:
			for char_num in range(len(word)-1):
				target = self.add_affinity(target, character_bbox[total_letters], character_bbox[total_letters+1])
				total_letters += 1
			total_letters += 1

		return target / 255, np.float32(target != 0)

	def __getitem__(self, item):

		image = plt.imread(self.base_path+'/'+self.imnames[item][0]).transpose(2, 0, 1)/255
		weight, target = self.generate_target(image.shape, self.charBB[item].copy())
		weight_affinity, target_affinity = self.generate_affinity(image.shape, self.charBB[item].copy(), self.txt[item].copy())

		return image, weight, target, weight_affinity, target_affinity

	def __len__(self):

		return len(self.imnames)


if __name__ == "__main__":

	dataloader = DataLoader('train')
	image, weight, target, weight_affinity, target_affinity = dataloader[0]

	plt.imsave('image.png', image.transpose(1, 2, 0))
	plt.imsave('target.png', target)
	plt.imsave('weight.png', weight)
	plt.imsave('weight_affinity.png', weight_affinity)
	plt.imsave('target_affinity.png', target_affinity)
	plt.imsave('together.png', np.concatenate([weight[:, :, None], weight