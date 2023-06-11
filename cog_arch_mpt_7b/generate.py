import torch as to

def generate( model, tokenizer, device, prompts ):
    used_input_text = prompts[ "solve_riddle/solve_riddle" ]
    encoded_ids  = tokenizer.encode( used_input_text, return_tensors = "pt" )
    input_ids = encoded_ids.to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids, use_cache = True, do_sample = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            # min_length = 400,
            max_new_tokens = 50, top_p = 0.95, top_k = 50, temperature = 0.8,
            repetition_penalty = 1.02
        )

    generated_ids = generated_ids[ 0 ]
    items = [ el.item() for el in generated_ids ]
    output_text = tokenizer.decode( items )
    
    return output_text.replace( used_input_text, "" ).strip()