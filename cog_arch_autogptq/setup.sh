cd /workspace

apt update -y
# apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

cd /workspace/miniconda3/envs
conda create -p cog_arch python=3.9 pip -y

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate cog_arch

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# conda install chardet cchardet -y
yes | pip install auto-gptq[triton]

cd /workspace/llms/cog_arch_autogptq
python start_loop.py