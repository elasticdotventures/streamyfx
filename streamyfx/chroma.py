
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np


def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle representing the center
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    # https://en.wikipedia.org/wiki/Image_moment
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{} {}x{}".format(i + 1, cY, cX), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	
	return (cnts, boundingBoxes)



#
# Brians custom chroma filter.
#
def transmogrify(frame, frame_x, frame_y, imgx): 

    image_copy = np.copy(frame)
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_copy);  plt.show();
        
    # CHROMA-BLUE
#    lower_blue = np.array([0, 0, 100])     ##[R value, G value, B value]
#    upper_blue = np.array([120, 100, 255]) 

    # CHROMA-GREEN
    # 24b24e or 0, 177, 64  -- 00b140
    offset = 0x4F # wow! jpeg really fucks with color! 
    lower_green = np.array([0,0xb1-offset,0])
    upper_green = np.array([offset, 0xb1+offset, 0x40+offset ])

    ## cv2.inRange finds pixels which are inbetween lower_green, upper_green
    mask = cv2.inRange(image_copy, lower_green, upper_green)

    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]        ## this sets the mask area, in the masked_image to zero (black)
    # draw_contour(masked_image)
    # plt.imshow(masked_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 
    # return(masked_image);	 ## UNCOMMENT TO RETURN MASKED IMAGE (chroma areas are black, before edge detection)
    
    edge_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)	# convert to grayscale to reduce # of layers to rpocess
    edge_image[mask == 0] = [0]        ## this sets everything NOT in the mask area to black (only chroma areas)

    # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    accumEdged = np.zeros(edge_image.shape[:2], dtype="uint8")
    # loop over the blue, green, and red channels, respectively  
    # (this takes about 30% of the time, it would be better to flatten the edge image to binary)
    for chan in cv2.split(edge_image):
    	# blur the channel, extract edges from it, and accumulate the set
    	# of edges for the image
    	chan = cv2.medianBlur(chan, 11)
    	edged = cv2.Canny(chan, 50, 200)
    	accumEdged = cv2.bitwise_or(accumEdged, edged)
    	# show the accumulated edge map
    	# cv2.imshow("Edge Map", accumEdged); plt.imshow(edge_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 

    # find contours in the accumulated image, keeping only the largest ones (RETR_EXTERNAL)
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)	 # this is a convenience function in imutil, verifies opencv countour signature.
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]   # cnts is 2d array of ??, keep top 5

    if len(cnts)==0:
        # cnts will be zero when no countours can be found (i.e. blank screen)
        return(frame);

    # Find contours for image, which will detect all the boxes
    #im2, contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours according to the provided method
    # Sort all the contours by top to bottom.
    # (contours, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")
    ## NOTE: i'm not sorting by screeen position ^^^ anymore. instead using the line below to get bounding Rectangles.
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # loop over the contours and draw them
    #orig = edge_image.copy();
    #for (i, c) in enumerate(cnts):
    #		orig = draw_contour(orig, c, i)
    
    # show the original, unsorted contour image
    # cv2.imshow("Unsorted", orig)
    # plt.imshow(edge_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 
    
    (x1,y1,wd,ht) = boundingBoxes[0];   ## grab the largest bounding box. 
    # print("chroma-area x1:{} y1:{} wd:{} ht:{}".format(x1,y1,wd,ht))

    ## create a 100% black image, and convert it to 
    ar_black = np.zeros([frame_y,frame_x,3],dtype=np.uint8)
    ar_black.fill(0)
    
    imgx = imgx.resize( (wd,ht), resample=Image.LANCZOS )       # TODO: increase magic i.e. how we resample, maintain aspect ratio. 

    img_black = Image.frombytes("RGB",(frame_x,frame_y),ar_black)    # convert to a PIL object
    img_black.paste( imgx, (x1,y1) )        ## I couldn't find a clean way to composite images into space, otherwise could skip steps.
    
    # cv2_background = cv2_background[0:frame_y, 0:frame_x]     # not needed. 
    cropmask_background = np.array(img_black)
    cropmask_background[mask == 0] = [0, 0, 0]      ## apply mask to crop_background

    final_image = cropmask_background + masked_image
    # plt.imshow(final_image); plt.show();
    return(final_image);



