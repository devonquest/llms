triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
model="mosaicml/mpt-7b-instruct"
attn_impl="triton"

apt update
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
pip install toolz transformers einops accelerate "$triton"

# python run_llm_transformers.py --model $model --attn_impl $attn_impl