cd /workspace

apt update -y
apt upgrade -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p /workspace/miniconda3
# repeat after restart
export PATH=$PATH:/workspace/miniconda3/bin

conda create --name autogptq python=3.9 pip -y
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate gptq
pip install auto-gptq[triton]

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
git clone https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit
cd GPT4-X-Alpaca-30B-4bit
git lfs pull --include="gpt4-x-alpaca-30b-4bit.safetensors"

cd /workspace/llms/cog_arch_gptq
python start_loop.py

# 

import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Download the model from HF and store it locally, then reference its location here:
quantized_model_repo = "MetaIX/GPT4-X-Alpaca-30B-4bit"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_repo, device="cuda:0", use_fast=True, cache_dir="/workspace/cache")

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=-1,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_repo, device="cuda:0", use_triton=False, use_safetensors=True,
    # low_cpu_mem_usage=True,
    use_cuda_fp16=True,
    torch_dtype=torch.float16, cache_dir="/workspace/cache",
    model_basename="gpt4-x-alpaca-30b-4bit",
    quantize_config=quantize_config
).to("cuda:0")

prompt = "Write a story about llamas"
prompt_template = f"### Instruction:\n\nSummarize in 4 words:\n\n{ text }\n\n\### Response:"

tokens = tokenizer(prompt, return_tensors="pt").to("cuda:0").input_ids
output = model.generate(input_ids=tokens, max_new_tokens=500, do_sample=True, temperature=0.6)
print(tokenizer.decode(output[0], skip_special_tokens=True))

"""
### Instruction:

Summarize the input text in 3 different lengths:

- 50 words
- 5 words
- a single word

### Input:

Tokyo's architecture is a mesmerizing blend of tradition and innovation,
showcasing the city's unique character. Walking through the bustling streets,
one is captivated by the seamless juxtaposition of ancient temples and
ultramodern skyscrapers. The iconic Tokyo Tower stands tall, reminiscent of
the Eiffel Tower, offering panoramic views of the city's architectural marvels.
The imperial palace, surrounded by lush gardens, presents a stunning display
of classical Japanese design, with its intricate wooden structures and tranquil
moats. Tokyo's skyline is dominated by sleek glass towers that house
futuristic offices and luxury apartments, reflecting the city's status as a
global economic powerhouse. The vibrant Shibuya crossing is not just a
bustling intersection but a testament to Tokyo's architectural innovation, with
its illuminated billboards and towering facades. Traditional wooden machiya
townhouses, with their distinctive latticed windows and tiled roofs, can still
be found tucked away in the historic neighborhoods, providing a glimpse into
Tokyo's past. Architectural wonders like the Meiji Shrine, with its majestic
torii gates and serene forest surroundings, offer a serene escape from the urban
chaos. Tokyo's architectural landscape constantly evolves, with cutting-edge
designs such as the iconic Tokyo Skytree, a towering communication tower that
boasts a futuristic aesthetic and serves as a symbol of Japan's technological
advancements. From ancient temples to modern skyscrapers, Tokyo's architecture
is a testament to the city's rich heritage and its unwavering embrace of
progress.

### Response:

# 50 words

Tokyo's architecture seamlessly blends tradition and innovation, showcasing
    ancient temples, ultramodern skyscrapers, iconic landmarks like Tokyo
    Tower and Meiji Shrine, and vibrant neighborhoods like Shibuya. The city's
    architectural marvels reflect its rich heritage and commitment to
    progress, making it a captivating global hub of design.

# 5 words

Traditional and modern Tokyo.

# a single word

Tokyo

### Instruction:

Summarize the input text in 3 different lengths:

- 50 words
- 5 words
- a single word

### Input:

In the heart of a bustling metropolis, a captivating tapestry of urban
life unfolds before your eyes. Majestic skyscrapers reach toward the heavens,
their sleek facades reflecting the vibrant energy that permeates the streets
below. The rhythm of life pulses through every corner, as a kaleidoscope of
diverse cultures intertwines, creating a vibrant mosaic of human experiences.
Sidewalks teem with a symphony of sounds, from the chatter of passersby to
the enticing melodies spilling out from cozy cafes and lively music venues.
The air is alive with the aromas of world cuisines, wafting from bustling food
carts and acclaimed restaurants, offering a tantalizing culinary journey for
adventurous palates. Vibrant street art adorns the walls, transforming once
ordinary spaces into canvases of creativity, telling stories of passion,
resilience, and the human spirit. Parks and green spaces provide a sanctuary
amidst the urban bustle, inviting city dwellers to reconnect with nature and
find solace in the midst of the concrete jungle. Technology hums in the
background, driving innovation and connecting people in ways never before
imagined. Markets bustle with activity, where the vibrant colors of fresh
produce, the intricate craftsmanship of local artisans, and the friendly
banter of vendors create a sensory feast for visitors and locals alike. In
this vibrant tapestry of urban life, dreams take flight, connections are
forged, and the city's pulse beats with endless possibilities. It is a place
where the past and the future converge, inviting all who enter to embrace the
energy, seize the opportunities, and create their own unique story within the
vibrant embrace of the metropolis.

### Response:
"""

# prompt = """### Instruction:

# Write a python function that counts the words in a txt file.
# '''

# ### Response:
# """

# prompt = """### Instruction:

# Look at the following riddle and tell me whether and how it is different than a riddle you know. Be brief.

# '''
# I'm standing at a river bank with a cabbage, a goat and a lion beside me and a boat in front of us.

# Rules:

# - leaving the lion and cabbage alone together is not allowed
# - leaving the lion and goat alone together is not allowed
# - the boat can carry me and only one additional passenger at each river crossing

# How do I get the lion, cabbage and goat safely across the river?
# '''

# ### Response:
# """

prompt = """### Instruction:

Look at the following riddle and note that it is a variation of the one with the fox, the hen and the bag of grains.
Recognize what's been changed and how your solution must change accordingly. Then provide the solution. Do not default
to known solutions, but reason from first principles, step by step.

'''
I'm standing at a river bank with a cabbage, a goat and a lion beside me and a boat in front of us.

Rules:

- leaving the lion and cabbage alone together is not allowed
- leaving the lion and goat alone together is not allowed
- the boat can carry only two entities during one river crossing, so me ( as I must steer the boat ) and one of the others

How do I get the lion, cabbage and goat safely across the river?
'''

### Response:
"""

tokens = tokenizer(prompt, return_tensors="pt").to("cuda:0").input_ids
output = model.generate(
    input_ids=tokens, max_new_tokens=400, do_sample=True, temperature=0.8,
    top_p=0.95,
    top_k=50,
    # repetition_penalty=1.02,
    # use_cache=True,
    # eos_token_id=tokenizer.eos_token_id,
    # pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))