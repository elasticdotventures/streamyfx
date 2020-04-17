ffmpeg build notes

# AOM
# PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_SHARED=on -DBUILD_SHARED_LIBS=1 -DENABLE_NASM=on ../aom && PATH="$HOME/bin:$PATH" make && make install

# optional flags for ffmpeg that need research:
#EXPORT CFLAGS=" -g -O2 -lm -ldl -Wall -Wpointer-arith -finline-functions -ffast-math -funroll-all-loops"


cd ~/ffmpeg_sources && \
# wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-libs="-lpthread -lm" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm -ldl -fPIC" \
  --bindir="$HOME/bin" \
  --enable-libaom \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree --enable-gpl \
  --enable-libzmq --enable-openssl --disable-gnutls --enable-hardcoded-tables \
  --enable-libopenh264  --enable-pic --enable-pthreads --enable-shared  \
  --enable-zlib --enable-version3 --enable-shared --enable-static &&
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r


#--prefix=/home/ubuntu/anaconda3 
#--cc=/home/conda/feedstock_root/build_artifacts/ffmpeg_1556785800657/_build_env/bin/x86_64-conda_cos6-linux-gnu-cc 
# --disable-doc --disable-openssl --enable-avresample --enable-gnutls --enable-gpl
# --enable-hardcoded-tables --enable-libfreetype --enable-libopenh264 --enable-libx264 --enable-pic 
# --enable-pthreads --enable-shared --enable-static --enable-version3 --enable-zlib --enable-libmp3lame
