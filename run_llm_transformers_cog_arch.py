import sys
import time as tm
import re

import toolz as tz

import torch as to
import transformers as tf

cache_dir = "./cache"

task = "- learning how to play piano"
generation_attempts = 0

def any( predicate, iterable ):
    len( list( filter( predicate, iterable ) ) ) > 0

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

def isKeyCarg( c ): c.startswith( "--" )

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

options = cargs_to_options( sys.argv[ 1: ] )
generate, tokenizer = \
    setup_pipeline( options[ "model" ], options[ "attn_impl" ] )

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

# - cognitive architecture based on prompting LLMs
#     - Perception
#     - Input Processing:
#         - Text Structuring
#         - Encoding for Interaction
#         - Prompt Design
#     - Working Memory Activation:
#         - Transfer Encoded Information
#         - Working Memory Processing:
#             - Storage and Retrieval
#             - Contextual Integration
#             - Temporary Computation
#             - Communication with Other Components
#     - Memory Index Creation:
#         - Analyzing Semantic Relationships
#         - Clustering Similar Concepts
#         - Identifying Key Themes or Categories
#     - Sparse Path Generation:
#         - Determining Possible Relevant Paths from Memory Index
#         - Assessing Relevancy of Paths to the Current Context or Query
#     - Reasoning and Knowledge Retrieval:
#         - Formulating Specific Prompts for the LLM
#         - Retrieving and Interpreting LLM Responses
#         - Integration with Decision-Making:
#             - Evaluation of Retrieved Information
#             - Consideration of Working Memory Contents
#     - Recursive Refinement:
#         - Assessing Satisfaction with Preliminary Decision
#         - If Unsatisfied, Return to Sparse Path Generation Stage
#     - Output Generation:
#         - Translate Decision into Output
#     - Long-Term Memory Interaction (interwoven throughout steps 3-8):
#         - Identify and Record Important or Relevant Information for Future
#             Reference
#         - Trigger 'Reset' to Memory Index Creation if Repeated Failure to
#             Satisfy
#     - Output Decision:
#         - Communicate Final Decision or Conclusion