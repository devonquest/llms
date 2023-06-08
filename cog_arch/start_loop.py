import importlib as il
import subprocess
from contextlib import ExitStack

import torch as to
import transformers as tf

import generate as gn

cache_dir = "../cache"

def git_pull():
    try:
        subprocess.check_output( ['git', 'pull'] )
        print( "\nGit pull successful!" )
    except subprocess.CalledProcessError as e:
        print( f"\nError: Git pull failed.\n\nMessage:\n\n{ e.output }" )

def create_pipeline( name, attn_impl, device, tokenizer ):
    config = tf.AutoConfig.from_pretrained(
        name, cache_dir = cache_dir, trust_remote_code = True
    )
    config.attn_config['attn_impl'] = attn_impl
    config.init_device = device

    return tf.pipeline(
        "text-generation",
        model = tf.AutoModelForCausalLM.from_pretrained(
            name, config=config, torch_dtype=to.bfloat16,
            trust_remote_code=True,
            device_map = "auto", cache_dir = cache_dir
        ),
        tokenizer = tokenizer
    )

def setup_pipeline( name, attn_impl ):
    device = "cuda:0"
    tokenizer = tf.AutoTokenizer.from_pretrained( name, cache_dir = cache_dir )

    return create_pipeline( name, attn_impl, device, tokenizer ), tokenizer

def load_prompts():
    with ExitStack() as stack:
        names = [ "summarize", "compress_4" ]
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

def loop_inference( generate, tokenizer ):
    global prompts

    user_msg = input(
        "\nOptions:"
        "\n\n- nothing to generate"
        "\n- r to pull repo"
        "\n- p to reload prompts"
        "\n- rp to join two previous options"
        "\n- anything else to exit"
        "\n\nType an option and press enter: "
    )

    if user_msg == "":
        print( "\nGenerating...\n" )
        response = il.reload( gn ).generate( generate, tokenizer, prompts )
        print( f"\n---\n\nResponse:\n\n{ response }\n\n---" )
    elif user_msg == "r":
        git_pull()
    elif user_msg == "p":
        prompts = load_prompts()
    elif user_msg == "rp":
        git_pull()
        prompts = load_prompts()
    else:
        return
    
    loop_inference( generate, tokenizer )

generate, tokenizer = setup_pipeline( "mosaicml/mpt-7b-instruct", "triton" )
loop_inference( generate, tokenizer )