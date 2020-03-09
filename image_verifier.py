# VERIFICATION MODULE
import torch
import numpy as np
from skimage.color import lab2rgb
from skimage.viewer import ImageViewer

data_path_save = 'data/test_data_set/test/'
# data_path_save = 'data/reduced/train/'


def reconstruct_predicted(image_gray, image_ab):

    img = np.zeros((256, 256, 3))
    img[:, :, 0] = image_gray
    img[:, :, 1:] = image_ab
    img = lab2rgb(img)  # (256, 256, 3), float(0., 1.)
    img = img * 255
    img = img.astype('uint8')  # (256, 256, 3), int(0, 255)
    return img


abval = torch.load(data_path_save + "abval_mushroom.pt")
lval = torch.load(data_path_save + "lval_mushroom.pt")
i = 1
img = reconstruct_predicted(lval[i], abval[i])
viewer = ImageViewer(img)
viewer.show()
