cd /workspace

apt update -y
# apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after reboot
export PATH=$PATH:/workspace/miniconda3/bin

conda create --name cog_arch python=3.9 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate cog_arch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
yes | pip install vim triton

git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
# pip uninstall quant-cuda -y
git stash
# toggle between branches
git checkout triton
# git checkout cuda
# git checkout fastest-inference-4bit
# git checkout old-cuda

sed -i "s/safetensors==0.3.0/safetensors==0.3.1/g" requirements.txt
yes | pip install -r requirements.txt
# toggle between branches
python setup_cuda.py install

# toggle between branches
sed -i "s/model.load_state_dict(safe_load(checkpoint))/model.load_state_dict(safe_load(checkpoint), strict=False)/g" llama.py
sed -i "s/model.load_state_dict(safe_load(checkpoint))/model.load_state_dict(torch.load(checkpoint), strict=False)/g" llama.py

cd /workspace
wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
bash ./script.deb.sh
apt install git-lfs
truncate -s 0 /etc/gitconfig

git lfs install --skip-smudge
# toggle between models
# git clone https://huggingface.co/TheBloke/gpt4-x-vicuna-13B-GPTQ
# git clone https://huggingface.co/TheBloke/wizardLM-7B-GPTQ
# git clone https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g
# git clone https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit
git clone https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ
cd Wizard-Vicuna-13B-Uncensored-GPTQ
git lfs pull --include="Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

cd /workspace/llms/cog_arch_gptq
python start_loop.py