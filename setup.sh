# git lfs install
# git clone "https://huggingface.co/$model"

# apt install -y vim python3.10-venv
# python -m venv ./

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
git checkout fastest-inference-4bit

pip install -r requirements.txt
python setup_cuda.py install

# for cmd in on off; do
#     cat << EOF | sed 's/^\s*//' > "$cmd"
#     #!/bin/bash

#     $([ "$cmd" = "on" ] && echo "source ./bin/activate" || echo "deactivate")
# EOF

#     chmod +x "$cmd"
# done

# source on

# # triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
# triton="triton"
# pip install disutils toolz transformers sentencepiece auto-gptq einops accelerate "$triton"
# # pip install toolz transformers einops accelerate

cd cog_arch_gptq
python start_loop.py