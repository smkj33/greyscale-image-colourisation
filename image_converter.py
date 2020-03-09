from PIL import Image

from skimage.color import rgb2lab
from skimage.transform import resize
import glob
import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Pytorch device: " + str(device))


data_path = 'data/test_data_set/test/'
data_path_save = 'data/test_data_set/test/'
# data_path = 'data/train_data_set/'
# data_path_save = 'data/reduced/train/'

lval = []
abval = []

# for filename in glob.glob(data_path + 'n12998815*.JPEG'):
for filename in glob.glob(data_path + '*.JPEG'):

    im = Image.open(filename)
    img_rgb = resize(np.asarray(im), (256, 256), anti_aliasing=True, mode='reflect')

    # img_lab = torch.from_numpy(rgb2lab(img_rgb)).type(torch.FloatTensor)
    # lval.append(img_lab[:, :, 0])
    # abval.append(img_lab[:, :, 1:])

    """
    VERIFICATION OF RGB AND LAB RANGES
    Following also verifies that normalisation is not required
    """
    # print(np.amax(np.amax(np.asarray(im)[:, :, 0:], axis=1), axis=0),
    #       np.amin(np.amin(np.asarray(im)[:, :, 0:], axis=1), axis=0))
    # print(np.amax(np.amax(rgb2lab(np.asarray(im))[:, :, 0:], axis=1), axis=0),
    #       np.amin(np.amin(rgb2lab(np.asarray(im))[:, :, 0:], axis=1), axis=0))
    # print(np.amax(np.amax(np.asarray(im)[:, :, 0:]/255, axis=1), axis=0),
    #       np.amin(np.amin(np.asarray(im)[:, :, 0:]/255, axis=1), axis=0))
    # print(np.amax(np.amax(rgb2lab(np.asarray(im)[:, :, 0:]/255), axis=1), axis=0),
    #       np.amin(np.amin(rgb2lab(np.asarray(im)[:, :, 0:]/255), axis=1), axis=0))


"""
File saving
"""
# torch.save(torch.stack(abval), data_path_save + "abval_mushroom.pt")
# torch.save(torch.stack(lval), data_path_save + "lval_mushroom.pt")
