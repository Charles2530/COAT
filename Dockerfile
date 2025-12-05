# --------------- 1. 基础镜像 ---------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# --------------- 2. 系统必备 ---------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --------------- 3. 安装 Miniconda ---------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# 验证 conda 安装并配置（跳过 conda update，避免网络问题）
RUN /opt/conda/bin/conda --version && \
    /opt/conda/bin/conda config --set always_yes true && \
    /opt/conda/bin/conda config --set changeps1 false && \
    /opt/conda/bin/conda config --set channel_priority flexible

# --------------- 4. 创建 coat 环境 ---------------
ENV COAT_ENV=coat
# 创建环境，使用 conda-forge 频道（更可靠）
RUN /opt/conda/bin/conda create -n $COAT_ENV python=3.10.14 -y -c conda-forge || \
    (echo "Conda create with conda-forge failed, trying defaults..." && \
    /opt/conda/bin/conda create -n $COAT_ENV python=3.10.14 -y -c defaults)
# 让后续 RUN 指令自动激活环境
SHELL ["conda", "run", "-n", "coat", "/bin/bash", "-c"]

# --------------- 5. 升级 pip / setuptools ---------------
RUN python -m pip install --upgrade pip setuptools wheel

# --------------- 6. 复制源码 & 安装 Python 包 ---------------
WORKDIR /workspace
COPY . /workspace

# 6-1. 安装 coat 本体
RUN pip install -e .

# 6-2. flash-attn（预编译 wheel 下载快，否则源码编 10 分钟）
RUN pip install flash-attn --no-build-isolation

# 6-3. 编译 FP8 optimizer
RUN cd coat/optimizer/kernels && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" python setup.py install

# 6-4. 编译 DeepGEMM
RUN cd third_party/DeepGEMM && \
    python setup.py install

# 6-5. 安装 OLMo
RUN cd examples/OLMo && \
    pip install -e .[all]

# --------------- 7. 清理 ---------------
RUN conda clean -afy && \
    rm -rf ~/.cache/pip

# --------------- 8. 入口 ---------------
# 容器启动后自动激活 coat 环境
ENV BASH_ENV ~/.bashrc
RUN echo "conda activate coat" >> ~/.bashrc
ENTRYPOINT ["conda", "run", "-n", "coat", "--no-capture-output"]
CMD ["/bin/bash"]