import torch as to

def generate( device, model, tokenizer, input_text ):
    input_ids = tokenizer.encode( input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids,
            use_cache=True, do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # min_length = 400,
            # max_length = 512,
            max_new_tokens = 512,
            top_p = 0.95,
            top_k = 50,
            temperature = 0.3,
            repetition_penalty=1.02
        )

    return tokenizer.decode( [ el.item() for el in generated_ids[ 0 ] ] )