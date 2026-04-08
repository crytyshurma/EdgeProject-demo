FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# python deps
RUN pip3 install --no-cache-dir \
    numpy \
    python-dotenv \
    loguru \
    opencv-python-headless

# copy project
COPY . /workspace

# create folders
RUN mkdir -p data/recordings data/snapshots

CMD ["python3", "main.py"]
