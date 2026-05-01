FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libfftw3-dev \
    libyaml-dev \
    libsamplerate0-dev \
    libtag1-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install jupyter notebook pandas matplotlib essentia

RUN pip3 install git+https://github.com/MTG/freesound-python.git

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
