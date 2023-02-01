FROM tensorflow/tensorflow:2.11.0-gpu
WORKDIR /work
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
