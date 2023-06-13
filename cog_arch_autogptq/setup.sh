cd /workspace

apt update -y
# apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

if [ ! -d "/workspace/miniconda3/envs" ]; then
    mkdir -p /workspace/miniconda3/envs
fi

# cd /workspace/miniconda3/envs
# conda deactivate
# conda remove -p cog_arch --all -y

cd /workspace/miniconda3/envs
conda create -p cog_arch python=3.9 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate cog_arch

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
yes | pip install sentencepiece triton

cd /workspace
git clone https://github.com/PanQiWei/AutoGPTQ.git
cd AutoGPTQ
yes | pip install .[triton]

cd /workspace/llms/cog_arch_autogptq
python start_loop.py