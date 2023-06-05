triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
model="mosaicml/mpt-7b-instruct"
attn_impl="triton"

pip install toolz transformers einops accelerate "$triton"

cd /workspace
python run_llm_transformers.py --model $model --attn_impl $attn_impl