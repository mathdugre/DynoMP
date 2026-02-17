ARG ORG=verificarlo
ARG VERIFICARLO_VERSION=v2.1.0
ARG PRISM_MODE=sr
ARG PRISM_DISPATCH=static
ARG PYTHON_MAJOR_VERSION=3.10
ARG PYTHON_MINOR_VERSION=19

# Build stage
FROM ${ORG}/verificarlo:${VERIFICARLO_VERSION} AS builder
ARG PYTHON_MAJOR_VERSION
ARG PYTHON_MINOR_VERSION

# Common environment variables
ENV PRISM_FLAGS="--prism-backend=sr --prism-backend-dispatch=static -march=native --verbose --inst-fma"
ENV VFC_BACKENDS="libinterflop_ieee.so"

# Setup build dependencies in a single layer
RUN apt-get update -qqq && \
    apt-get install -y --no-install-recommends -qqq \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget \
    curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev wget fort77 gfortran cmake lzma-dev liblzma-dev \
    libmpfr6 libmpfr-dev pybind11-dev python3-pybind11 && \
    mkdir -p /opt/build/

# 3. Build Python 3.10 From Source (The robust fix)
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}/Python-${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}.tgz && \
    tar xvf Python-${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}.tgz && \
    cd Python-${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j $(nproc) && \
    make altinstall && \
    # Create symlinks so python work as expected
    ln -sf /usr/local/bin/python${PYTHON_MAJOR_VERSION} /usr/bin/python${PYTHON_MAJOR_VERSION} && \
    ln -sf /usr/local/bin/python${PYTHON_MAJOR_VERSION} /usr/bin/python3 && \
    ln -sf /usr/local/bin/python${PYTHON_MAJOR_VERSION} /usr/bin/python && \
    # Install build tools
    python${PYTHON_MAJOR_VERSION} -m pip install --upgrade pip setuptools wheel typing_extensions cmake ninja

# Copy necessary resources
COPY fuzzy/docker/resources/lapack/blas-sanity-check.sh /tmp/blas-sanity-check.sh
RUN chmod +x /tmp/blas-sanity-check.sh
COPY fuzzy/docker/resources/lapack/test_blas.c /tmp/test_blas.c
COPY fuzzy/docker/resources/pytorch/test_fuzzy_pytorch.py /tmp/test_fuzzy_pytorch.py
COPY vprec-pytorch-exclude.txt /tmp/pytorch-vfc-exclude.txt

# Build Lapack
RUN cd /opt/build/ && \
    wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz && \
    tar xf v3.12.1.tar.gz && \
    cd /opt/build/lapack-3.12.1/ && \
    mkdir build && \
    cd /opt/build/lapack-3.12.1/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCBLAS=ON -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_FLAGS="${PRISM_FLAGS}" \
    -DCMAKE_Fortran_FLAGS="${PRISM_FLAGS}" \
    -DCMAKE_C_COMPILER=verificarlo-c \
    -DCMAKE_Fortran_COMPILER=verificarlo-f \
    .. && \
    make -j $(nproc) && \
    make install && \
    cd /tmp/ && ./blas-sanity-check.sh

# Install patched verificarlo
RUN cd /opt/build && \
    git clone --depth=1 -b v2.1.0 https://github.com/verificarlo/verificarlo.git && \
    cd verificarlo && \
    sed -i 's/prism_fatal_error("Function not found: " + functionName);/return PrismFunction(function, passing_style);/' src/libvfcinstrumentprism/libVFCInstrumentPRISM.cpp && \
    ./autogen.sh && \
    ./configure --with-llvm=$(llvm-config-7 --prefix) && \
    make -C src/libvfcinstrumentprism && \
    make -C src/libvfcinstrumentprism install

# Build PyTorch -- inference version was 2.2.1 for training we are working with 2.6.0
RUN cd /opt/build/ && \
    git clone --depth=1 -b v2.2.1 https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive && \
    pip install "numpy<2" && \
    pip install -r requirements.txt && \
    sed -i '/Module(Module&&)/s/noexcept//g' torch/csrc/jit/api/module.h && \
    mkdir -p build && \
    cd build && \
    cmake -GNinja \
    -DCMAKE_INSTALL_PREFIX="/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/torch" \
    -DPYTHON_EXECUTABLE="/usr/bin/python${PYTHON_MAJOR_VERSION}" \
    -DCMAKE_MODULE_PATH="${PWD}/cmake/public" \
    -DCMAKE_CXX_COMPILER=verificarlo-c++ \
    -DCMAKE_CXX_FLAGS="${PRISM_FLAGS} --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    -DCMAKE_C_COMPILER=verificarlo-c \
    -DCMAKE_C_FLAGS="${PRISM_FLAGS} --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    -DCMAKE_ASM_COMPILER=verificarlo-c \
    # -DSTATIC_DISPATCH_BACKEND=CPU \
    # -DTP_BUILD_PYTHON=ON \
    -DTP_INSTALL_LIBDIR="/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/torch/lib" \
    # -DBUILD_BINARY=OFF \
    # -DUSE_DISTRIBUTED=ON \
    -DUSE_CUDA=OFF \
    -DUSE_MKLDNN=OFF \
    -DUSE_NUMA=OFF \
    -DUSE_FBGEMM=ON \
    -DUSE_LAPACK=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBLAS_LIBRARIES="/usr/local/lib/libcblas.so" \
    -DBLAS=Generic \
    -DWITH_BLAS=generic \
    -DUSE_NATIVE_ARCH=ON \
    -DUSE_NUMPY=ON \
    -DBUILD_CAFFE2=ON \
    -DBUILD_CAFFE2_OPS=ON \
    # -DBUILD_ONNX_PYTHON=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    ..

RUN cd /opt/build/pytorch/build && \
    ninja && \
    ninja install && \
    cd /opt/build/pytorch && \
    CC=verificarlo-c \
    CXX=verificarlo-c++ \
    LDSHARED=verificarlo-c \
    CFLAGS="-L/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/torch/lib ${PRISM_FLAGS} --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    LDFLAGS="-L/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/torch/lib -shared ${PRISM_FLAGS} --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    python3 setup.py install

RUN cd /tmp/ && python3 test_fuzzy_pytorch.py

ENV PYTHONPATH=/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages


RUN python${PYTHON_MAJOR_VERSION} -c "import torch; print('Torch successfully loaded: {torch.__version__}, found at:, {torch.utils.cmake_prefix_path}')"

# Build torchaudio
RUN cd /opt/build/ && \
    git clone --depth=1 -b v2.2.1 https://github.com/pytorch/audio.git && \
    cd audio && \
    python${PYTHON_MAJOR_VERSION} -m pip install ffmpeg sentencepiece deep-phonemizer soundfile sox && \
    CC=verificarlo-c CXX=verificarlo-c++ \
    CFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    CXXFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    LDFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma -shared --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
    LDSHARED=verificarlo-c \
    python${PYTHON_MAJOR_VERSION} -m pip install --no-build-isolation . -r requirements.txt


# # Build torchvision
# RUN cd /opt/build/ && \
#     git clone --depth=1 -b v0.17.2 https://github.com/pytorch/vision.git && \
#     cd vision && \
#     python${PYTHON_MAJOR_VERSION} -m pip install setuptools wheel && \
#     CC=verificarlo-c CXX=verificarlo-c++ \
#     CFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
#     CXXFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
#     LDFLAGS="--prism-backend=sr --prism-backend-dispatch=static --verbose --inst-fma -shared --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
#     LDSHARED=verificarlo-c \
#     python${PYTHON_MAJOR_VERSION} -m pip install --no-build-isolation . 

# # # Build torchaudio
# # RUN cd /opt/build/ && \
# #         git clone --depth=1 -b v2.2.1 https://github.com/pytorch/audio.git && \
# #         cd audio && \
# #         python${PYTHON_MAJOR_VERSION} -m pip install ffmpeg sentencepiece deep-phonemizer soundfile sox && \
# #         CC=verificarlo-c CXX=verificarlo-c++ \
# #         CFLAGS="--prism-backend=ud --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
# #         CXXFLAGS="--prism-backend=ud --prism-backend-dispatch=static --verbose --inst-fma --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
# #         LDFLAGS="--prism-backend=ud --prism-backend-dispatch=static --verbose --inst-fma -shared --exclude-file=/tmp/pytorch-vfc-exclude.txt" \
# #         LDSHARED=verificarlo-c \
# #         python${PYTHON_MAJOR_VERSION} -m pip install . -r requirements.txt

# # Runtime stage
# FROM ${ORG}/verificarlo:${VERIFICARLO_VERSION}
# ARG PYTHON_MAJOR_VERSION
# ARG PYTHON_MINOR_VERSION

# # Copy only necessary files from builder
# COPY --from=builder /usr/local/lib/ /usr/local/lib/
# COPY --from=builder /usr/local/include/ /usr/local/include/
# COPY --from=builder /usr/local/bin/ /usr/local/bin/
# COPY --from=builder /usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/ /usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages/
# COPY --from=builder /opt/build/pytorch/build/.ninja_log /tmp/pytorch_build.log

# # Set environment variables
# ENV VFC_BACKENDS="libinterflop_ieee.so"
# ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
# ENV PYTHONPATH="/usr/local/lib/python${PYTHON_MAJOR_VERSION}/site-packages:${PYTHONPATH}"

# # RUN pip install torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-deps

# CMD ["/bin/bash"]