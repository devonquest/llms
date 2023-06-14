import torch as to

def generate( device, model, tokenizer, prompts ):
    # used_input_text = prompts[ "summarize" ]
    used_input_text = """Within this decade, AI will become a ubiquitous tool for businesses,
transforming the way they operate and compete. It will enable companies to
automate routine tasks, optimize processes, and make better decisions based on
data and insights. However, AI also raises significant challenges, including
the need to manage and govern data, ensure transparency and accountability, and
address potential job displacement and societal impacts. To harness the
potential of AI while mitigating its risks, businesses and governments must
work together to establish a framework for responsible AI. This framework
should include principles for fairness, transparency, accountability, and
inclusivity,as well as mechanisms for oversight, regulation, and collaboration.
By working together, we can ensure that AI delivers on its promise to improve
our lives and our world. Furthermore, """

    input_ids = tokenizer.encode( used_input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids,
            use_cache = True,
            do_sample = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            max_new_tokens = 300,
            top_p = 0.1,
            top_k = 50,
            temperature = 0.1,
            # repetition_penalty = 1.02
        )

    output_text = tokenizer.decode( [ el.item() for el in generated_ids[ 0 ] ] )
    return output_text.replace( "<s>", "" ).replace( "</s>", "" ) \
        .replace( used_input_text, "" ).strip()