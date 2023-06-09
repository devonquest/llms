import torch as to

def generate( model, tokenizer, device, prompts ):
    used_input_text = prompts[ "solve_riddle/0" ]
    encoded_ids  = tokenizer.encode( used_input_text, return_tensors = "pt" )
    input_ids = encoded_ids.to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids, use_cache = True, do_sample = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            # min_length = 400,
            max_new_tokens = 1000, top_p = 0.1, top_k = 50, temperature = 0.1,
            repetition_penalty = 1.02
        )

    generated_ids = generated_ids[ 0 ]
    items = [ el.item() for el in generated_ids ]
    output_text = tokenizer.decode( items, skip_special_tokens = True )
    
    return output_text.replace( used_input_text, "" ).strip()

    # output_text = tokenizer.decode( items )
    # print( output_text )
    # return ""