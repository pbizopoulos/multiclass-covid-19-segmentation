FROM tensorflow/tensorflow:2.10.0-gpu
WORKDIR /work
COPY pyproject.toml .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir .
