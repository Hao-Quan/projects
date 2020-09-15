import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

img = np.array(Image.open('1.png'))

#img = mpimg.imread('1.png')
imgplot = plt.imshow(img)

plt.savefig("10.png", dpi=1000)
plt.show()