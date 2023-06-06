import sys
import time as tm
import re

import toolz as tz

import torch as to
import transformers as tf

def any( predicate, iterable ):
    len( list( filter( predicate, iterable ) ) ) > 0

def isKeyCarg( c ): c.startswith( "--" )

prev_prompt = ""

def create_pipeline( name, attn_impl, device, tokenizer ):
    config = tf.AutoConfig.from_pretrained( name, trust_remote_code = True )
    config.attn_config['attn_impl'] = attn_impl
    config.init_device = device

    return tf.pipeline(
        "text-generation",
        model = tf.AutoModelForCausalLM.from_pretrained(
            name, config=config, torch_dtype=to.bfloat16,
            trust_remote_code=True,
            device_map = "auto"
        ),
        tokenizer = tokenizer
    )

def setup_pipeline( name, attn_impl ):
    device = "cuda:0"
    tokenizer = tf.AutoTokenizer.from_pretrained( name )

    return create_pipeline( name, attn_impl, device, tokenizer ), tokenizer

def count_tokens( message ):
    words = re.split( r'\s+', message )
    num_tokens = len( words ) / 0.75

    return int( num_tokens )

def measure( inferred_text, before, after ):
    num_tokens = count_tokens( inferred_text )

    return num_tokens, num_tokens / ( after - before )

def generate_once( prompt, max_new_tokens, generate, tokenizer ):
    before = tm.time()
    response = generate(
        prompt,
        temperature=0.8, top_p=0.95, top_k=50, max_new_tokens=max_new_tokens,
        use_cache=True, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.02
    )

    after = tm.time()
    inferred_response = response[ 0 ][ "generated_text" ].replace( prompt, "" )

    return inferred_response, *measure( inferred_response, before, after )

def loop_input_generation( generate, tokenizer ):
    prompt_msg, max_msg = [
        input( msg ) for msg in [ "\nEnd or prompt: ", "Max new tokens: " ]
    ]
    lowered_prompt_msg = prompt_msg.lower()
    max_new_tokens = 0 if (
        max_msg == "" or not max_msg.isdigit()
    ) else int( max_msg )

    if lowered_prompt_msg == "end":
        return
    elif max_new_tokens < 20 or max_new_tokens > 1000:
        print( "\nInvalid max new tokens. Valid range: 0 < x < 1000" )
        
        return

    global prev_prompt

    prompt = prompt_msg if lowered_prompt_msg != "prev" else prev_prompt
    if prompt == "":
        return
    
    prev_prompt = prompt

    response, num_tokens, tps = generate_once(
        prompt, max_new_tokens, generate, tokenizer
    )

    print(
        f"\n---\n\nResponse:\n\n{ response }"
        + f"\n\nNum tokens: { num_tokens }\nt/s: { tps }"
    )

    loop_input_generation( generate, tokenizer )

def print_cargs_config( names, defaults ):
    defaults_msg = "\n---\n\nConfig:\n"
    for n, d in zip( names, defaults ):
        defaults_msg += f"\nKey: { n }\t, Value: { d }"

    print( defaults_msg )

def cargs_to_options( cargs ):
    options = { "model": "mosaicml/mpt-7b-instruct", "attn_impl": "triton" }
    valid_attn_impls = "torch", "triton"

    if len( cargs ) == 0: return options
    
    grouped = tz.groupby( lambda e: e[ 0 ] % 2 == 0, enumerate( cargs ) )
    if len( grouped ) % 2 != 0:
        print( "Error: Uneven number of command line arguments." )
        sys.exit()
    else:
        keys, values = \
            tz.map( lambda g: list( tz.map( tz.second, g ) ), grouped.values() )

        if any( lambda s: not isKeyCarg( s ), keys ) \
                or any( isKeyCarg, values ) :
            print( "Error: Wrong order of command line arguments." )
            sys.exit()
        else:
            keys = map( lambda k: k.replace( "--", "" ), keys )
            for k, v in zip( keys, values ):
                if k not in options:
                    print( f"Error: Wrong command line argument key: { k }" )
                    sys.exit()
                elif k == "attn_impl" and v not in valid_attn_impls:
                    print(
                        f"Error: Wrong command line argument value: { v }. " \
                        + f"Valid: { valid_attn_impls }"
                    )
                    sys.exit()
                else:
                    options[ k ] = v

    return options

options = cargs_to_options( sys.argv[ 1: ] )
generate, tokenizer = \
    setup_pipeline( options[ "model" ], options[ "attn_impl" ] )

def make_breakdown_prompt( task ):
    return f"""
    ### Instruction:

    Break down the task '{ task }' into a flat list. Include _just_ the list
    in your response.

    ### Response:
    """

generation_attempts = 0

def generate_with_condition( prompt, predicate ):
    global generation_attempts
    response, *_ = generate_once( prompt, 1000, generate, tokenizer )
    lines = response.split( "\n" )

    for l in lines:
        if not predicate( l ) and generation_attempts < 5:
            generation_attempts += 1
            print( "Malformed response. Trying again." )

            return generate_with_condition( predicate )
    
    generation_attempts = 0
    return response, lines

depth = 0
task = "- learning how to play piano"
prompt = make_breakdown_prompt( task )

outline, lines = \
    generate_with_condition( prompt, lambda l: l.startswith( "-" ) )
new_outline = [ f"{ task }" ] + []

print( outline )
print( new_outline )