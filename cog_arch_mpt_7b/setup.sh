cd /workspace

apt update -y
apt upgrade -y
apt install build-essential clang -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

cd /workspace/miniconda3/envs
conda create --prefix cog_arch python=3.9 pip -y

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate cog_arch

conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia -y
triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
yes | pip install transformers einops "$triton"

cd /workspace/llms/cog_arch_mpt_7b
python start_loop.py