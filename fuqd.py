# https://github.com/kkroening/ffmpeg-python/tree/master/examples#audiovideo-pipeline\

import ffmpeg
import numpy as np

def process_frame_simple(frame):
    '''Simple processing example: darken frame.'''
    return frame * 0.3

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
def process_chroma(frame, frame_x,frame_y): 

    # image = cv2.imread(frame)		# uses file!
    #image = cv2.imdecode(frame,3)
    #frame_y, frame_x, frame_colors = image.shape

    image = frame;
    return(image);		## short circuit, can't make the rest of this code work. (unsure of image format)
    
    # IMG_COMPOSITE_FILENAME = "D:\\sp_globalwarming498x370.jpg";
    IMG_COMPOSITE_FILENAME = "./samples/67899726_10215481153205503_6083579797621964800_n.jpg"
    IMG_COMPOSITE_FILENAME = "./samples/68555334_10162036457415177_5374707312411803648_n.jpg"
    
    # print('Image type: ', type(image), 'Image Dimensions : ', image.shape)


    image_copy = np.copy(image)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_copy);  plt.show();

        
    lower_blue = np.array([0, 0, 100])     ##[R value, G value, B value]
    upper_blue = np.array([120, 100, 255]) 

    # 24b24e or 0, 177, 64  -- 00b140
    offset = 0x4F # wow! jpeg really fucks with color! 
    lower_green = np.array([0,0xb1-offset,0])
    upper_green = np.array([offset, 0xb1+offset, 0x40+offset ])

    ## cv2.inRange finds pixels which are inbetween lower_green, upper_green
    mask = cv2.inRange(image_copy, lower_green, upper_green)

    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]        ## this sets the mask area, in the masked_image to zero. 
    # draw_contour(masked_image)
    # plt.imshow(masked_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 

    edge_image = np.copy(image_copy)
    edge_image[mask == 0] = [0, 0, 0]        ## this sets the mask area, in the masked_image to zero. 


    # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    accumEdged = np.zeros(edge_image.shape[:2], dtype="uint8")
    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(edge_image):
    	# blur the channel, extract edges from it, and accumulate the set
    	# of edges for the image
    	chan = cv2.medianBlur(chan, 11)
    	edged = cv2.Canny(chan, 50, 200)
    	accumEdged = cv2.bitwise_or(accumEdged, edged)
    	# show the accumulated edge map
    	# cv2.imshow("Edge Map", accumEdged); plt.imshow(edge_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 


    # find contours in the accumulated image, keeping only the largest ones
    cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]   # cnts is 2d array of ?? 

    if len(cnts)==0:
        print("skipping\n");
        return(frame);


    # Find contours for image, which will detect all the boxes
    #im2, contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")
    # print(boundingBoxes) boundingBox is x1,y1,length,height

    # loop over the (unsorted) contours and draw them
    #orig = edge_image.copy();
    #for (i, c) in enumerate(cnts):
    #	orig = draw_contour(orig, c, i)
    
    # show the original, unsorted contour image
    # cv2.imshow("Unsorted", orig)
    # plt.imshow(edge_image); plt.show();      ## at this point, masked_image has the green mask removed, set to zero. 
    
    (x1,y1,wd,ht) = boundingBoxes[0];   ## grab the largest bounding box. 
    print("chroma-area x1:{} y1:{} wd:{} ht:{}".format(x1,y1,wd,ht))

    ## create a 100% black image, and convert it to 
    ar_black = np.zeros([frame_y,frame_x,3],dtype=np.uint8)
    ar_black.fill(0)
    img_black = Image.frombytes("RGB",(frame_x,frame_y),ar_black)    # convert to a PIL object

    # imgx = Image.open("D:\\equations720x720.jpg")               # source image we'll resize into the bounding box. 
    imgx = Image.open(IMG_COMPOSITE_FILENAME)               # source image we'll resize into the bounding box. 
    imgx = imgx.resize( (wd,ht), resample=Image.LANCZOS )       # TODO: add magic here like how we resample, maintain aspect ratio. 
    img_black.paste( imgx, (x1,y1) )        ## I couldn't find a clean way to composite images into space, otherwise could skip steps.
    
    # cv2_background = cv2_background[0:frame_y, 0:frame_x]     # not needed. 
    cropmask_background = np.array(img_black)
    cropmask_background[mask == 0] = [0, 0, 0]      ## apply mask to crop_background

    final_image = cropmask_background + masked_image
    # plt.imshow(final_image); plt.show();
    return(final_image);



in_filename = 'greenscreen.mkv';
out_filename = 'out.avi';
width = 1280;
height = 720;

process1 = (
    ffmpeg
    .input(in_filename)
    #.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=8)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    .output(out_filename, pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
        
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )	 # used by example.

    # See examples/tensorflow_stream.py:
    #out_frame = deep_dream.process_frame(in_frame)		# doesn't work  (yet)
    #out_frame = process_frame_simple(in_frame)		# works, makes frame darker.
    
    # chroma system works on images, not frames. used imread
    pc_frame = ( np.frombuffer(in_bytes, np.uint8) );
    out_frame = process_chroma(pc_frame, width,height )
    
    # out_frame = in_frame;

    process2.stdin.write(
        out_frame
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process1.wait()
process2.wait()











class DeepDream(object):
    '''DeepDream implementation, adapted from official tensorflow deepdream tutorial:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
    Credit: Alexander Mordvintsev
    '''

    _DOWNLOAD_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    _ZIP_FILENAME = 'deepdream_model.zip'
    _MODEL_FILENAME = 'tensorflow_inception_graph.pb'

    @staticmethod
    def _download_model():
        logger.info('Downloading deepdream model...')
        try:
            from urllib.request import urlretrieve  # python 3
        except ImportError:
            from urllib import urlretrieve  # python 2
        urlretrieve(DeepDream._DOWNLOAD_URL, DeepDream._ZIP_FILENAME)

        logger.info('Extracting deepdream model...')
        zipfile.ZipFile(DeepDream._ZIP_FILENAME, 'r').extractall('.')

    @staticmethod
    def _tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See `_resize` function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    @staticmethod
    def _base_resize(img, size):
        '''Helper function that uses TF to resize an image'''
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]

    def __init__(self):
        if not os.path.exists(DeepDream._MODEL_FILENAME):
            self._download_model()

        self._graph = tf.Graph()
        self._session = tf.InteractiveSession(graph=self._graph)
        self._resize = self._tffunc(np.float32, np.int32)(self._base_resize)
        with tf.gfile.FastGFile(DeepDream._MODEL_FILENAME, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self._t_input = tf.placeholder(np.float32, name='input') # define the input tensor
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self._t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})

        self.t_obj = self.T('mixed4d_3x3_bottleneck_pre_relu')[:,:,:,139]
        #self.t_obj = tf.square(self.T('mixed4c'))

    def T(self, layer_name):
        '''Helper for getting layer output tensor'''
        return self._graph.get_tensor_by_name('import/%s:0'%layer_name)

    def _calc_grad_tiled(self, img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = self._session.run(t_grad, {self._t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def process_frame(self, frame, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(self.t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self._t_input)[0] # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = frame
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[:2]
            lo = self._resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self._resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = self._resize(img, hi.shape[:2])+hi
            for i in range(iter_n):
                g = self._calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
                #print('.',end = ' ')
        return img

