import tables
import torch
import numpy as np
import matplotlib.pyplot as plt


def print_single_image(imgid, dataname, phase):
    """prints a single image from a pytable

    Args:
        imgid (int): id of the image in the pytable
        dataname (str): name of the dataset
        phase (str): phase of the data

    Returns: 
        NOTHING
    """

    db = tables.open_file(f"./{dataname}_{phase}.pytable")
    img = db.root.imgs[imgid, ::]
    label = torch.tensor(np.array(db.root.labels[imgid]))

    # img = img[:,:,None].repeat(3,axis=2) #convert to 3 channel, only if images are stored as single-channel.

    plt.title(label)
    plt.imshow(img)


def print_bag(dataname, phase, bag_size, figsize=(20,20), child_directory=""):
	"""prints an entire bag of images from a pytable

	Args:
		dataname (str): name of the dataset
		phase (str): phase of the data
		bag_size (int): number of items to be printed
		figsize (int tuple): size of each figure

	Returns: 
		NOTHING
	"""

	db = tables.open_file(f".{child_directory}/{dataname}_{phase}.pytable")
	cols = 3

	for i in range(0, bag_size, cols):
		fig, ax = plt.subplots(1, cols, figsize=figsize)
		ax = ax.flatten()

		for j in range(0, cols):
			if i + j > bag_size:
				break

			label = np.array(db.root.labels[i + j]).item(0)
			slide_id = np.array(db.root.slide_ids[i+j]).item(0)

			ax[j].set_title(f"label: {label}, slide_id: {slide_id}")
			ax[j].imshow(db.root.imgs[i + j, ::])

# a similar method will be needed upon generating the bag_labels.csv file.
def fname2fnumber(fname):
	"""Converts the name of a WSI to a unique number mapped to the slide-level label
	
		Args:
			fname (str): name of the file 
	
		Returns: 
			int[]: [slide_id, instance_#]
	"""