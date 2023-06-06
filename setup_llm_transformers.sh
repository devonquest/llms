triton="triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
model="mosaicml/mpt-7b-instruct"
attn_impl="triton"

apt install python3.10-venv
python -m venv myenv
source myenv/bin/activate

pip install toolz transformers einops accelerate "$triton"
# python run_llm_transformers.py --model $model --attn_impl $attn_impl