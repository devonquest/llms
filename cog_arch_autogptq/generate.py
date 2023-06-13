import torch as to

def generate( device, model, tokenizer, prompts ):
    used_input_text = prompts[ "summarize" ]
    input_ids = tokenizer.encode( used_input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids, use_cache = True, do_sample = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            # min_length = 400,
            max_new_tokens = 50,
            top_p = 0.95, top_k = 50, temperature = 0.8,
            num_beams = 1,
            repetition_penalty=1.02
        )

    output_ids = generated_ids[ 0 ]
    id_items = [ el.item() for el in output_ids ]

    output_text = tokenizer.decode( id_items, skip_special_tokens = True )
    return output_text.replace( "<s>", "" ).replace( "</s>", "" ) \
        .replace( used_input_text, "" ).strip()