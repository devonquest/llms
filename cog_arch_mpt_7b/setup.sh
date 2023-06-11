cd /workspace

apt update -y
apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

conda create --name mpt-7b python=3.9 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate mpt-7b

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
yes | pip install transformers einops

cd /workspace/llms/cog_arch_mpt_7b
python start_loop.py