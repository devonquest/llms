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
        names = [ "summarize", "improve", "solve_riddle" ]

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

def init_model( device ):
    name = "mosaicml/mpt-7b-instruct"

    config = tf.AutoConfig.from_pretrained( name, trust_remote_code = True )
    config.attn_config[ "attn_impl" ] = "triton"
    config.init_device = device

    model = tf.AutoModelForCausalLM.from_pretrained(
        name, config = config, torch_dtype = to.bfloat16,
        trust_remote_code = True
    ).to( device )
    tokenizer = tf.AutoTokenizer.from_pretrained( "EleutherAI/gpt-neox-20b" )

    return model, tokenizer

def generate_timed( model, tokenizer, device, prompts, input_text ):
    global gn

    print( "\nGenerating...\n" )
    gn = il.reload( gn )

    before = tm.time()
    output_text = gn.generate( model, tokenizer, device, prompts, input_text )
    after = tm.time()

    num_tokens, tps = measure_tokens( output_text, before, after )

    print( f"\n---\n\nResponse:\n\n{ output_text }\n\n---" )
    print( f"\nNum tokens: { num_tokens }\ttps: { tps }\n\n---" )

def loop_inference( model, tokenizer, device, prompts, input_text ):
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

    generate_timed( model, tokenizer, device, prompts, input_text )
    loop_inference( model, tokenizer, prompts, input_text )

device = "cuda:0"
model, tokenizer = init_model( device )

loop_inference(
    model, tokenizer, device, prompts, "Within this decade, AI will"
)