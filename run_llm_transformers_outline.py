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

task = "- learning how to play piano"
generation_attempts = 0

def make_breakdown_prompt( task, context ):
    return f"""### Q1:
Can you break down the task "- learn how to play the piano" into a flat list
of 2 to 7 items given the context below?

Context:
- learning how to play piano

### A1:
- Master piano keys and basic exercises
- Understand music theory and practice songs
- Maintain daily routine and seek regular feedback

### Q2:
Can you break down the task "{ task }" into a flat list of 2 to 7 items
given the context below?

Context:
{ context }

### A2:
"""

def generate_with_predicate(prompt, predicate):
    global generation_attempts
    response, *_ = generate_once(prompt, 1000, generate, tokenizer)
    
    if not predicate(response):
        generation_attempts += 1
        if generation_attempts == 10:
            print("Generation attempts exhausted, exiting...")
            exit(1)
        print(f"\nInvalid response:\n\n{response}")
        return generate_with_predicate(prompt, predicate)

    return response

def is_flat_list(response):
    lines = response.split('\n')
    num_usable_lines = 0
    for line in lines:
        if line.startswith("-"):
            num_usable_lines += 1
        elif num_usable_lines > 0:
            num_usable_lines = 0
    return num_usable_lines >= 2

def sanitize_flat_list(response):
    lines = response.split('\n')
    sanitized = [line for line in lines if line.startswith("-")]
    return '\n'.join(sanitized)

def item_from_path( outline, path ):
    next, *path_tail = path
    item = outline[ next ]

    if path_tail:
        return item_from_path( item, path_tail )
    else:
        return item
    
def set_item_at_path( outline, path, element ):
    next, *path_tail = path
    cur_element = outline[ next ]

    if path_tail:
        set_item_at_path( cur_element, path_tail, element )
    else:
        outline[ next ] = element

def break_down(task, context):
    global generation_attempts

    prompt = make_breakdown_prompt(task, context)
    response = generate_with_predicate(prompt, is_flat_list)
    generation_attempts = 0

    return sanitize_flat_list(response)

def outline_to_string(outline, depth=0):
    result = ""
    tab = "  " * (depth + 1)

    for item in outline:
        if isinstance(item, dict):
            for task, subtasks in item.items():
                result += f"{tab}{task}\n"
                result += outline_to_string(subtasks, depth + 1)
        else:
            result += f"{tab}{item}\n"

    return result

# TODO: Consider removing.
#   - new break_down_deep with context token length limitation
#       def _break_down_deep(
#           task,
#           depth,
#           cur_depth=0,
#           outline=[],
#           outline_path=[0],
#           cur_line_i=0
#       ):
#           - if cur_depth < depth
#               - break down task, passing outline as context ( returns the block of subtasks as a string )
#               - split the subtasks into lines
#               - for each i, line in enumerate( lines )
#                   - _break_down_deep(
#                       line,
#                       depth,
#                       cur_depth + 1,
#                       if item at outline_path of outline doesn't exist or is not dict
#                           outline with item at outline_path set to a dict with
#                               the task as key and the lines as the value,
#                       else
#                           outline
#                       outline_path + task + i,
#                       i
#                     )
#           - return with outline

def _break_down_deep(
    task, depth, cur_depth = 0, outline = [], outline_path = [ 0 ],
):
    if cur_depth < depth:
        children = break_down( task, outline_to_string( outline ) )
        lines = children.split( "\n" )

        if not outline:
            outline.append( { task: lines } )
        else:
            if not isinstance( item_from_path( outline, outline_path ), dict ):
                set_item_at_path( outline, outline_path, { task: lines } )

        for i, line in enumerate( lines ):
            _break_down_deep(
                line, depth, cur_depth + 1, outline, outline_path + task + i
            )
        
    return outline

# TODO: Consider removing.
# def break_down_deep(task, depth):
#     return task + "\n" + outline_to_string( _break_down_deep(task, depth) )

print( _break_down_deep( "- developing a mobile app for a restaurant", 1 ) )