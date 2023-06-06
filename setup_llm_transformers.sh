triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
model="mosaicml/mpt-7b-instruct"
attn_impl="triton"

pip install venv
pip venv myenv
source myenv/bin/activate
# pip install toolz transformers einops accelerate "$triton"

# python run_llm_transformers.py --model $model --attn_impl $attn_impl
pip install toolz