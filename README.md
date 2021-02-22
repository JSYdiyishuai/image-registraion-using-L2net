# image registration and blend 

Firstly, use SIFT or ORB, SURF to extract key points and descriptors for registration.

Secondly, divide the image into 32*32 small patches and use L2-Net to generate descriptors for registration. (gray images)

Reference link: https://github.com/zhaobenx/Image-stitcher
		https://github.com/virtualgraham/L2-Net-Python-Keras

Y. Tian, B. Fan, F. Wu. "L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space", CVPR, 2017.

# Usages

modify the image directory, run `stitch.py`

