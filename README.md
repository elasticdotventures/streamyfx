# streamyfx

rocket's video processing tools (experimental)

so the challenge of realtime stream transformation is 


FFMPEG terminology:
DEMUXER are configured elements in FFmpeg that READ the multimedia streams from a particular type of file (or stream)
MUXERS are configured elements in FFmpeg which WRITE the multimedia streams to a particular type of file (or stream)
https://www.ffmpeg.org/ffmpeg-formats.html

RawVideo DROPS audio
https://github.com/stoyanovgeorge/ffmpeg/wiki/Encode-Raw-Video
Y4M - Y4M contains information about the resolution of the picture, scan type, frame rate and the color space,
YUV - does NOT contain any meta-data, it's just raw frames in rgb24 means RGB = 3 uint8 bytes

rawvideo -  This demuxer allows one to read raw video data. Since there is no header specifying the assumed video parameters, the user must specify them in order to be able to decode the data correctly.
	framerate : Set input video frame rate. Default value is 25.
	pixel_format : Set the input video pixel format. Default value is yuv420p but rgb24 is easier to process
	video_size : Set the input video size i.e. 320x240 This value must be specified explicitly.

audio notes
aac  - Advanced Audio Coding (AAC) encoder, preferred format for 
opus is opensource, lossy, but effective at low bitrates
wavpack is lossless but doesn't appear to be streamable?
vorbis is open, lossy, but high performing

# see list of pix_fmts and formats
ffmpeg -formats
ffmpeg -pix_fmts

# transcode to vp9 webm - works
ffmpeg -i ../rm.mkv -c:v libvpx-vp9 -pass 1 -b:v 10000K -threads 8 -speed 4 -an output.webm

# y4m to fifo => flv+rtmp - works
ffmpeg -y -i ../rm.mkv videofifo.y4m
ffmpeg -i videofifo.y4m -f flv rtmp://localhost:1935/mytv/test

# create two files (mkv and aac) - WORKS!
ffmpeg -y -i ../rm.mkv -c:v copy -map 0:0 output.mkv -c:a copy -map 0:1 output.aac
ffmpeg -y -i ../rm.mkv -c:v copy -map 0:0 output.mp4 -c:a copy -map 0:1 output.wav
ffmpeg -i output.mkv -i output.aac -f flv rtmp://localhost:1935/mytv/test

# OR merge
ffmpeg -i videofifo.mkv -i audiofifo.aac -f flv rtmp://localhost:1935/mytv/test

# 

# so the problem with using two pipes is a deadlock 
ffmpeg -y -i ../rm.mkv -vcodec rawvideo videofifo.y4m  -acodec aac audiofifo.aac

# note: fifo's are faster on /dev/shm



# split using 'tee' output (DOES NOT WORK)
https://trac.ffmpeg.org/wiki/Creating%20multiple%20outputs
ffmpeg -y -i ../rm.mkv -map 0:v -map 0:a -c:v libx264 -c:a aac -f tee "[select=\'v:0,a\':f=flv]output.mkv|[select=\'v:0,a\':f=flv]rtmp://localhost:1935/mytv/test"

mkfifo audiofifo.aac
mkfifo videofifo.y4m
ffmpeg -y -i ../rm.mkv -map 0:v -map 0:a -c:v libx264 -c:a aac -f tee "[select=\'v:0\']videofifo.y4m|[select=\'a:0\']audiofifo.aac"

# WRITING TO TWO NAMED PIPES: (DOES NOT WORK)
ffmpeg -y -i ../rm.mkv -map 0:v -map 0:a -c:v libx264 -c:a aac -f tee "[select=\'v:0\':f=flv]videofifo.y4m"
ffmpeg -i videofifo.y4m -map 0:v -vcodec libx264 -f flv rtmp://localhost:1935/mytv/test

ffmpeg -y -loglevel verbose -re -i ../rm.mkv -map 0:v -codec data -f tee  "[select=\'v:0\':codec=data:format=rawvideo:pix_format=rgb24]videofifo.y4m"


-format='rawvideo', pix_fmt='rgb24'
 format='rawvideo', pix_fmt='rgb24', s='{}x{}'

# split two files (DOES NOT WORK)
ffmpeg -i input-video.avi -vn -acodec copy output-audio.aac
ffmpeg -y -i ../rm.mkv -c:v copy -map 0:0 html/rm-video.mkv -c:a copy -map 0:1 html/rm-audio.aac
ffmpeg -y -i ../rm.mkv -c:v copy -s 1920x1080 -map 0:0 -f mpegts zmq:tcp://127.0.0.1:5555 -c:a copy -map 0:1 html/rm-audio.aac
ffmpeg -y -i ../rm.mkv -c:v copy -s 1920x1080 -map 0:0 -f mpegts zmq:tcp://127.0.0.1:5555 -c:a copy -map 0:1 -f wav zmq:tcp://127.0.0.1:5556


# combine two files
ffmpeg -i INPUT.mp4 -i AUDIO.wav -shortest -c:v copy -c:a aac -b:a 256k OUTPUT.mp4
ffmpeg -i zmq:tcp://localhost:5555 -map 0:0 -i azmq:tcp://localhost:5556 -shortest -c:v copy -c:a aac -b:a 256k html/test.mkv


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
