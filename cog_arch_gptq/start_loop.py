import sys
import importlib as il
import subprocess as sp
import contextlib as cl

import time as tm
import re

import torch as to
import transformers as tf

sys.path.append( "/workspace/GPTQ-for-LLaMa/" )
# toggle between branches
# lm = il.import_module( "llama_inference" )
lm = il.import_module( "llama" )
gn = il.import_module( "generate" )

def git_pull():
    try:
        sp.check_output( ['git', 'pull'] )
        print( "\nGit pull successful." )
    except sp.CalledProcessError as e:
        print( f"\nError: Git pull failed.\n\nMessage:\n\n{ e.output }" )

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

def generate_timed( device, model, tokenizer, input_text ):
    global gn

    print( "\nGenerating...\n" )
    gn = il.reload( gn )

    before = tm.time()
    output_text = gn.generate( device, model, tokenizer, input_text )
    num_tokens, tps = measure_tokens( output_text, before, tm.time() )

    print( f"\n---\n\nResponse:\n\n{ output_text }\n\n---" )
    print( f"\nNum tokens: { num_tokens }\ttps: { tps }\n\n---" )

def loop_inference( device, model, tokenizer, input_text ):
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

    generate_timed( device, model, tokenizer, input_text )
    loop_inference( device, model, tokenizer, input_text )

device = to.device( "cuda:0" )
model_dir = "/workspace/Wizard-Vicuna-13B-Uncensored-GPTQ"

# toggle between branches
model = lm.load_quant(
    model_dir,
    f"{ model_dir }/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
    4,
    128,
    act_order = False
).to( device )
# model = lm.load_quant(
#     model_dir,
#     f"{ model_dir }/gpt4-x-alpaca-30b-4bit.safetensors",
#     4,
#     -1,
#     "cuda:0"
# ).to( device )

tokenizer = tf.AutoTokenizer.from_pretrained( model_dir, use_fast = False )

loop_inference( device, model, tokenizer, "Within this decade, AI will" )