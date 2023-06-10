import importlib as il
import subprocess as sp
import contextlib as cl

import time as tm
import re

import torch as to
import transformers as tf

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

device_name = "cuda:0"
device = to.device( device_name )

model_repo = "roneneldan/TinyStories-33M"
tokenizer_repo = "EleutherAI/gpt-neo-125M"

config = tf.AutoConfig.from_pretrained( model_repo )
# config.attn_config[ "attn_impl" ] = "triton"
config.init_device = device_name

model = tf.AutoModelForCausalLM.from_pretrained( model_repo, config = config ) \
    # .to( device )
tokenizer = tf.AutoTokenizer.from_pretrained( "EleutherAI/gpt-neo-125M" )

input_text = "Once upon a time there was a pumpkin. It was a very special " \
    "pumpkin, it could speak. It was sad because it couldn’t move. Every day" \
    ", it would say"
loop_inference( device, model, tokenizer, "Mary said:" )