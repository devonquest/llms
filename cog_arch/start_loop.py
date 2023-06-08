import subprocess

import torch as to
import transformers as tf

cache_dir = "./cache"

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

def loop_inference( generate, tokenizer ):
    user_msg = input(
        "\nOptions:\n\n- enter to generate\n- pull to update prompt" \
        "\n- anything else to exit\n\nType an option: "
    )

    if user_msg == "":
        with open( "./prompt.txt", "r" ) as f:
            print( "\nGenerating...\n" )

            prompt = f.read()
            from generate import generate_with_predicate
            response = generate_with_predicate(
                prompt, lambda _: True, generate, tokenizer
            )

            print( f"\n---\n\nResponse:\n\n{ response }\n\n---" )
    elif user_msg == "pull":
        git_pull()
    else:
        return
    
    loop_inference()

generate, tokenizer = setup_pipeline( "mosaicml/mpt-7b-instruct", "triton" )
loop_inference( generate, tokenizer )