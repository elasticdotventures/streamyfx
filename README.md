# streamyfx

rocket's video processing tools (experimental)

streamyfx/chroma.py : contains the 'transmogrify' function that:
* detects largest chroma green area in an image (video-frame)
* scales the image to the largest chroma-box
* inserts the image onto a black frame in the space of the chroma area  
* overlays the video frame with the scaled chroma 

fuqd.py : contains a script for frame-by-frame video processing through ffmpeg named pipes
test.py : does a single image (greenscreen.jpg) into a html/ directory (for testing)

ffmpeg.md contains instructions for compiling ffmpeg with zeromq (zmq)
support

experiments in realtime video processing.
note: currently this process still dumps the audio.

next steps:
* convert so ffmpeq is using zmq sockets
