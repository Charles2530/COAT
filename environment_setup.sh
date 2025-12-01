#!/usr/bin/env bash
# conda activate /mnt/lm_data_afs/wangzining/charles/miniconda3/envs/coat
CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools

# install coat
pip install -e .
pip install -U flash-attn --no-build-isolation

# install fp8 optimizer
cd coat/optimizer/kernels/
TORCH_CUDA_ARCH_LIST="8.9 9.0" python setup.py install

# install DeepGEMM
cd third_party/DeepGEMM
python setup.py install


cd examples/OLMo/
pip install -e .[all]
cd ../..
# python -m build
# twine upload dist/*
