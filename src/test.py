import numpy as np

from src.data import Data
from skimage import io, filters, feature,transform
import matplotlib.pyplot as plt

import os

data = Data()

x,y=data.get_data(data.TEST_DIR)
print(y)
# img = io.imread(os.path.join(os.path.join(data.CROP_DIR, '00000'), '00000_00001.ppm'))
# print(type(img))
# img=cv2.imread(os.path.join(os.path.join(data.ORIGINAL_DIR, '00000'), '00000_00001.ppm'))
# print(type(img))
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.subplot(1, 1, 1)
# plt.imshow(img)
# plt.show()

# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# plt.imshow(hsv)
# plt.show()

# edges=cv2.Canny(img, 200, 400)

# edges = feature.canny(img, sigma=1)
# plt.imshow(edges)
# plt.show()

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# plt.imshow(closed)
# plt.show()

# edges = filters.sobel(img)
# img=transform.resize(img,(32,32))
# edges=feature.canny(img,sigma=1)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(edges)
# plt.show()


