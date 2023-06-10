import time as tm
import re

import importlib as il
import subprocess
from contextlib import ExitStack

import transformers as tf
import auto_gptq as aq

import generate as gn

cache_dir = "../cache"

def git_pull():
    try:
        subprocess.check_output( ['git', 'pull'] )
        print( "\nGit pull successful." )
    except subprocess.CalledProcessError as e:
        print( f"\nError: Git pull failed.\n\nMessage:\n\n{ e.output }" )

def get_config( has_desc_act ):
    return aq.BaseQuantizeConfig(
        bits = 4, group_size = 128, desc_act = has_desc_act
    )

def get_model( hf_repo, name, has_desc_act, triton ):
    model_suffix = ".compat.act-order"
    device = "cuda:0"

    # f"/workspace/llms/cache/models--{ hf_repo.replace( '/', '--' ) }"

    return aq.AutoGPTQForCausalLM.from_quantized(
        hf_repo, model_basename = f"{ name }{ model_suffix }",
        device = device,
        init_device = device,
        # device_map = "auto",
        use_safetensors = True,
        use_triton = triton,
        quantize_config = get_config( has_desc_act )
    )

def create_pipeline( hf_repo, name ):
    tokenizer = tf.AutoTokenizer.from_pretrained(
        hf_repo, cache_dir = cache_dir, use_fast = False
    )

    generate = tf.pipeline(
        "text-generation",
        model = get_model( hf_repo, name, False, True ),
        tokenizer = tokenizer
    )

    return generate, tokenizer

def load_prompts():
    with ExitStack() as stack:
        names = [ "summarize", "improve" ]

        return dict(
            zip(
                names,
                [
                    stack.enter_context( open( f"./prompts/{ n }.txt" ) ).read()
                    for n in names
                ]
            )
        )
    
prompts = load_prompts()

def reload_prompts():
    global prompts

    prompts = load_prompts()
    print( "\nPrompts reloaded." )

def pull_and_load():
    git_pull()
    reload_prompts()

def count_words( text ):
    words = re.split( r'\s+', text )

    return len( words )

def count_tokens( text ):
    num_tokens = count_words( text ) / 0.75

    return int( num_tokens )

def measure_tokens( text, before, after ):
    num_tokens = count_tokens( text )

    return num_tokens, num_tokens / ( after - before )

def generate_timed():
    print( "\nGenerating...\n" )

    before = tm.time()
    response = il.reload( gn ).generate( generate, tokenizer, prompts )
    num_tokens, tps = measure_tokens( response, before, tm.time() )

    print( f"\n---\n\nResponse:\n\n{ response }\n\n---" )
    print( f"\nNum tokens: { num_tokens }\ttps: { tps }\n\n---" )

def loop_inference( generate, tokenizer ):
    global prompts

    user_msg = input(
        "\nOptions:"
        "\n\n- r to pull repo and reload prompts"
        "\n- end to exit"
        "\n- anything else to generate"
        "\n\nType an option and press enter: "
    )

    if user_msg == "r":
        pull_and_load()
    elif user_msg == "end":
        return

    generate_timed()
    loop_inference( generate, tokenizer )

tf.logging.set_verbosity( tf.logging.CRITICAL )

import os
print( os.getcwd() )

generate, tokenizer = create_pipeline(
    "TheBloke/gpt4-x-vicuna-13B-GPTQ", "GPT4-x-Vicuna-13B-GPTQ-4bit-128g"
)
loop_inference( generate, tokenizer )