FROM tensorflow/tensorflow:2.13.0rc2-gpu
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
