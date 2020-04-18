import streamyfx.chroma as chroma;
import cv2;
import numpy as np;
import imutils;
from PIL import Image


TEST_INPUT_FILE = "greenscreen.jpg";
TEST_OUTPUT_FILE = "html/output.jpg";	

image = cv2.imread(TEST_INPUT_FILE)		# loads file!
image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

IMG_COMPOSITE_FILENAME = "./samples/67899726_10215481153205503_6083579797621964800_n.jpg"
imgx = Image.open(IMG_COMPOSITE_FILENAME)               # source image we'll resize into the bounding box. 
# print('Image type: ', type(image), 'Image Dimensions : ', image.shape)	# requires imtutils

image_copy = chroma.transmogrify(image_copy, 1280, 720, imgx)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

cv2.imwrite(TEST_OUTPUT_FILE, image_copy);

# image = cv2.imdecode(frame,3)
#frame_y, frame_x, frame_colors = image.shape


