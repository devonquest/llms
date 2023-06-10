cd /workspace

apt update -y
apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
export PATH=$PATH:/workspace/miniconda3/bin # repeat after restart

conda create --name gptq python=3.9 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
git checkout fastest-inference-4bit

sed -i "s/safetensors==0.3.0/safetensors==0.3.1/g" requirements.txt
pip install -r requirements.txt
python setup_cuda.py install

cd /workspace
wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
bash ./script.deb.sh
apt install git-lfs

git lfs install
git clone https://huggingface.co/TheBloke/gpt4-x-vicuna-13B-GPTQ

cd llms/cog_arch_gptq
python start_loop.py