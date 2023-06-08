# git lfs install
# git clone "https://huggingface.co/$model"

apt install -y vim python3.10-venv
python -m venv ./

for cmd in on off; do
    cat << EOF | sed 's/^\s*//' > "$cmd"
    #!/bin/bash

    $([ "$cmd" = "on" ] && echo "source ./bin/activate" || echo "deactivate")
EOF

    chmod +x "$cmd"
done

source on

triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
pip install toolz transformers einops accelerate "$triton"

# attn_impl="triton"
# python run_llm_transformers.py --model $model --attn_impl $attn_impl