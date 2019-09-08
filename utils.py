import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import animation
from IPython.display import display, HTML
#plt.rcParams['animation.ffmpeg_path'] = '/home/fabio/anaconda3/bin/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = '/home/fabio/anaconda3/envs/ai/bin/ffmpeg'


def plot_batch_mp4(video_array, std=None, mean=None):
	for video in video_array:
		plot_movie_mp4(video, std=std, mean=mean)

def plot_labeled_batch_mp4(video_array, labels, std=None, mean=None):
	for video, label in zip(video_array, labels):
		plot_movie_mp4(video, label, std=std, mean=mean)


def plot_movie_mp4(image_array, text=None, std=None, mean=None):
	if std and mean:
		tensor_list = torch.tensor(image_array.permute(1,0,2,3))
		for t, m, s in zip(image_array, mean, std):
			t.mul_(s).add_(m)
		image_array = torch.clamp(tensor_list.permute(1,2,3,0), 0, 1).numpy()
	else:
		image_array = image_array.permute(0,2,3,1).numpy()
	dpi = 60.0
	xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
	fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi);
	im = plt.figimage(image_array[0]);

	def animate(i):
		im.set_array(image_array[i])
		return (im)

	anim = animation.FuncAnimation(fig, animate, frames=len(image_array))

	if text:
		prepend = '<p>'+ str(text) +'</p>'
		display(HTML(prepend + anim.to_html5_video()))
	else:
		display(HTML(anim.to_html5_video()))

def calculate_loss_and_accuracy(validation_loader, model, criterion, stop_at = 1200, print_every=99999):
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	sz = len(validation_loader)
	
	for images, labels in validation_loader:
	
		if total%print_every == 0 and total > 0:
			accuracy = 100 * correct / total
			print(accuracy)
		
		if total >= stop_at:
			break;
		if torch.cuda.is_available():
			images = images.cuda()
			labels = labels.cuda()

		# Forward pass only to get logits/output
		outputs = model(images)
		
		#Get Loss for validation data
		loss = criterion(outputs, labels)
		total_loss += loss.item()


		# Get predictions from the maximum value
		_, predicted = torch.max(outputs.data, 1)

		# Total number of labels
		total += labels.size(0)
		steps += 1

		correct += (predicted == labels).sum().item()

		del outputs, loss, _, predicted

	accuracy = 100 * correct / total
	return total_loss/steps, accuracy

def calculate_loss_and_tops(validation_loader, model, criterion, stop_at = 1200, print_every=99999):
	correct, correct3, correct5 = 0, 0, 0
	total = 0
	steps = 0
	total_loss = 0
	sz = len(validation_loader)
	model.eval()

	for images, labels in validation_loader:

		if total%print_every == 0 and total > 0:
			top1 = 100 * correct / total
			top3 = 100 * correct3 / total
			top5 = 100 * correct5 / total
			print(top1, top3, top5)

		if total >= stop_at:
			break;
		if torch.cuda.is_available():
			images = images.cuda()
			labels = labels.cuda()

		# Forward pass only to get logits/output
		outputs = model(images)

		#Get Loss for validation data
		loss = criterion(outputs, labels)
		total_loss += loss.item()


		# Get predictions from the maximum value
		_, predicted = torch.max(outputs.data, 1)
		_, predicted3 = torch.topk(outputs.data, 3)
		_, predicted5 = torch.topk(outputs.data, 5)
		
		top3 = [l in p for (l,p) in zip(labels, predicted3)]
		top5 = [l in p for (l,p) in zip(labels, predicted5)]

		# Total number of labels
		total += labels.size(0)
		steps += 1

		correct += (predicted == labels).sum().item()
		correct3 += np.sum(top3)
		correct5 += np.sum(top5)
		

		del outputs, loss, _, predicted

		top1 = 100 * correct / total
		top3 = 100 * correct3 / total
		top5 = 100 * correct5 / total

	return total_loss/steps, top1, top3, top5

