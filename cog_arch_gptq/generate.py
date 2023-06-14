import torch as to

def generate( device, model, tokenizer, prompts ):
    # used_input_text = prompts[ "summarize" ]
    # used_input_text = "Within this decade, AI will"
    used_input_text = """### Instruction:

Summarize the input text in 3 different lengths:

- 50 words
- 5 words
- a single word

### Input:

The art of culinary excellence.

### Response:
    """
    input_ids = tokenizer.encode( used_input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids,
            use_cache=True, do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # min_length = 400,
            # max_length = 512,
            max_new_tokens = 50,
            top_p = 0.1,
            top_k = 50,
            temperature = 0.1,
            repetition_penalty=1.02
        )

    output_text = tokenizer.decode( [ el.item() for el in generated_ids[ 0 ] ] )
    return output_text.replace( "<s>", "" ).replace( "</s>", "" ) \
        .replace( used_input_text, "" ).strip()