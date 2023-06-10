cd /workspace

apt update -y
apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

conda create --name gptq python=3.10 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate gptq
conda install transformers triton

cd /workspace
wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
bash ./script.deb.sh
apt install git-lfs

git lfs install
# toggle between models
git clone https://huggingface.co/roneneldan/TinyStories-33M

cd /workspace/llms/cog_arch_standard
python start_loop.py