from torchvision.transforms import *
import numbers
import random
from PIL import Image

class GroupToTensor(object):
	def __init__(self):
		pass

	def __call__(self, img_group):
		return [ToTensor()(img) for img in img_group]
	
class GroupCenterCrop(object):
	def __init__(self, size):
		self.size = size

	def __call__(self, img_group):
		return [CenterCrop(self.size)(img) for img in img_group]
	
class GroupResize(object):
	def __init__(self, size):
		self.size = size

	def __call__(self, img_group):
		return [Resize(self.size)(img) for img in img_group]
	
class GroupResizeFit(object):
	def __init__(self, size):
		self.size = size

	def __call__(self, img_group):
		print(img_group[0].size)
		return [Resize(self.size)(img) for img in img_group]
	
class GroupExpand(object):
	def __init__(self, size):
		self.size = size

	def __call__(self, img_group):
		w, h = img_group[0].size
		tw, th = self.size
		out_images = list()
		if(w >= tw and h >= th):
			assert img_group[0].size == self.size
			return img_group
		for img in img_group:
			new_im = Image.new("RGB", (tw, th))
			new_im.paste(img, ((tw-w)//2, (th-w)//2))
			out_images.append(new_im)
		assert out_images[0].size == self.size
		return out_images

class GroupRandomCrop(object):
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size

	def __call__(self, img_group):

		w, h = img_group[0].size #120, 160
		th, tw = self.size #100, 140
		out_images = list()
		if (w - tw) < 0:
			print('W < TW')
			for img in img_group:
				new_im = Image.new("RGB", (tw, th))
				new_im.paste(img_group[0], ((tw-w)//2, (th-w)//2))
				out_images.append(new_im)
			return out_images

		x1 = random.randint(0, (w - tw))
		y1 = random.randint(0, (h - th))
		for img in img_group:
			if w == tw and h == th:
				out_images.append(img)
			else:
				out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

		return out_images

class GroupRandomRotation(object):
	def __init__(self, max):
		self.max = max

	def __call__(self, img_group):
		angle = random.randint(-self.max, self.max)
		return [functional.rotate(img, angle) for img in img_group]
	
class GroupNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor_list):
		# TODO: make efficient
		for t, m, s in zip(tensor_list, self.mean, self.std):
			t.sub_(m).div_(s)

		return tensor_list

class GroupUnormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor_list):
		# TODO: make efficient
		for t, m, s in zip(tensor_list, self.mean, self.std):
			t.mul_(s).add_(m)

		return tensor_list