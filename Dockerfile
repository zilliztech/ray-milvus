# syntax=docker/dockerfile:1.4
# Build ray-milvus with milvus-storage Python bindings

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl git g++ gcc make ccache \
    python3 python3-pip python3-venv build-essential libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget -qO- "https://cmake.org/files/v3.27/cmake-3.27.5-linux-$(uname -m).tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

# Install Conan
RUN pip3 install --no-cache-dir conan==1.61.0

# Setup Conan profile and remote
RUN conan profile new default --detect || true \
    && conan profile update settings.compiler.libcxx=libstdc++11 default
RUN conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --insert || true

# Set ccache configuration
ENV CCACHE_DIR=/root/.ccache
ENV PATH=/usr/lib/ccache:$PATH

WORKDIR /workspace

# Copy everything
COPY . .

# Initialize git submodules
RUN git config --global --add safe.directory /workspace && \
    git config --global --add safe.directory /workspace/milvus-proto && \
    git config --global --add safe.directory /workspace/milvus-storage && \
    git submodule update --init --recursive

# Build milvus-storage Python library (with cache)
# Note: Need to ensure conan remote is configured when using cache mount
RUN --mount=type=cache,target=/root/.conan \
    --mount=type=cache,target=/root/.ccache \
    conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --insert 2>/dev/null || true && \
    cd milvus-storage/cpp && make python-lib

# Create output directories and copy artifacts
RUN mkdir -p /workspace/libs && \
    cp milvus-storage/cpp/build/Release/libmilvus-storage.so /workspace/libs/

# Copy dependent libraries from Conan cache
RUN --mount=type=cache,target=/root/.conan \
    cp $(find /root/.conan/data/glog -name "libglog.so.1" | head -1) /workspace/libs/ && \
    cp $(find /root/.conan/data/gflags -name "libgflags_nothreads.so.2.2" | head -1) /workspace/libs/

# ==================== Runtime Stage ====================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip python3-venv liblzma5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy libraries from builder
COPY --from=builder /workspace/libs/ /app/libs/

# Copy Python modules
COPY --from=builder /workspace/milvus-storage/python/milvus_storage /app/milvus_storage/
COPY --from=builder /workspace/ray_milvus /app/ray_milvus/

# Create symlink for library discovery
RUN mkdir -p /app/milvus_storage/lib && \
    ln -s /app/libs/libmilvus-storage.so /app/milvus_storage/lib/libmilvus-storage.so

# Set library path and preload liblzma for glog
ENV LD_LIBRARY_PATH=/app/libs
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/liblzma.so.5

# Install Python dependencies
RUN pip3 install --no-cache-dir cffi pyarrow numpy ray pandas

ENV PYTHONPATH=/app

CMD ["python3"]
