FROM tensorflow/tensorflow:2.10.0-gpu
ENV PIP_NO_CACHE_DIR=1
WORKDIR /work
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .
