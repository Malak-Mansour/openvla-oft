# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation #if gives an error, do the following export statements to get CUDA (nvcc) path

# conda install nvidia/label/cuda-12.4.0::cuda-toolkit
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CONDA_PREFIX/bin:$PATH
# export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```