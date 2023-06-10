import torch as to

def generate( device, model, tokenizer, input_text ):
    input_ids = tokenizer.encode( input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample = True,
            min_length = 20,
            max_length = 100,
            top_p = 0.95,
            temperature = 0.6,
        )

    return tokenizer.decode( [ el.item() for el in generated_ids[ 0 ] ] )