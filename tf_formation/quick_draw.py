import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_dir = "quick_draw_dataset"
files = os.listdir(data_dir)
max_size_per_cl = 1500

draw_class = []

# Evalueate the size of the dataset
size = 0
i = 0
for name in files:
    draws = np.load(os.path.join(data_dir, name))
    # print("file shape: ", draws.shape)
    draws = draws[:max_size_per_cl] # Take only max_size_per_cl draw
    size += draws.shape[0]
    i += 1
# print("Images took  %s" % (i * max_size_per_cl))
# print("sise %s" % size)
images = np.zeros((size, 28, 28))
targets = np.zeros((size,))

it = 0
t = 0
raw_im = []
for name in files:
    # Open each dataset and add the new class
    draw_class.append(name.replace("full_numpy_bitmap_", "").replace(".npy", ""))
    draws = np.load(os.path.join(data_dir, name))
    draws = draws[:max_size_per_cl] # Take only max_size_per_cl draw
    # Add images to the buffer
    images[it:it+draws.shape[0]] = np.invert(draws.reshape(-1, 28, 28))
    targets[it:it+draws.shape[0]] = t
    # Iter
    it += draws.shape[0]
    t += 1
    raw_im.append(draws)
print(raw_im[0][0])
# plt.imshow(raw_im[0][0], cmap='binary')
# plt.show()
"""
images = images.astype(np.float32)
    
# Shuffle dataset
indexes = np.arange(size)
np.random.shuffle(indexes)
images = images[indexes]
targets = targets[indexes]

im_train, im_test, tar_train, tar_test = train_test_split(images, targets, test_size=0.2, random_state=1)

print("images.shape", im_train.shape)
print("targets.shape", tar_train.shape)

print("images_valid.shape", im_test.shape)
print("targets_valid.shape", tar_test.shape)

print(draw_class)"""