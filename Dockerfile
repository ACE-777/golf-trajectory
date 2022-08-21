FROM docker.io/nvidia/cuda:11.7.0-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        build-essential \
        curl \
        libpq-dev \
        nasm \
        git \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-venv \
        ninja-build \
        libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /golf

# Install requirements
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir -U pip==21.3.1 setuptools==59.6.0 wheel==0.37.1

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY magnus.py .
COPY curve.py .
COPY app.py .

RUN adduser --system --group golf
RUN chown -R golf:golf /golf
USER golf