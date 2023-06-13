import sys
import importlib as il
import subprocess as sp
import contextlib as cl

import time as tm
import re

import torch as to
import transformers as tf

sys.path.append( "/workspace/AutoGPTQ/" )
ag = il.import_module( "auto_gptq" )

gn = il.import_module( "generate" )

def git_pull():
    try:
        sp.check_output( ['git', 'pull'] )
        print( "\nGit pull successful." )
    except sp.CalledProcessError as e:
        print( f"\nError: Git pull failed.\n\nMessage:\n\n{ e.output }" )

def setup_model():
    repo = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
    cache_dir = "/workspace/cache"

    device = "cuda:0"

    tokenizer = tf.AutoTokenizer.from_pretrained(
        repo, cache_dir = cache_dir, device = device, use_fast = True
    )

    model = ag.AutoGPTQForCausalLM.from_quantized(
        repo,
        model_basename = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g" \
            ".compat.no-act-order",
        cache_dir = cache_dir,
        quantize_config = ag.BaseQuantizeConfig(
            bits = 4,
            group_size = 128,
            desc_act = False,
        ),
        device = device, use_triton = False,
        use_safetensors = True, use_cuda_fp16 = False,
        # torch_dtype = to.float16, 
    ).to( device )

    return model, tokenizer, device

def load_prompts():
    with cl.ExitStack() as stack:
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

def generate_timed( device, model, tokenizer ):
    global gn

    print( "\nGenerating...\n" )
    gn = il.reload( gn )

    before = tm.time()
    output_text = gn.generate( device, model, tokenizer, prompts )
    after = tm.time()

    num_tokens, tps = measure_tokens( output_text, before, after )

    print( f"\n---\n\nResponse:\n\n{ output_text }\n\n---" )
    print( f"\nNum tokens: { num_tokens }\ttps: { tps }\n\n---" )

def loop_inference( device, model, tokenizer ):
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

    generate_timed( device, model, tokenizer )
    loop_inference( device, model, tokenizer )

model, tokenizer, device = setup_model()
loop_inference( device, model, tokenizer )