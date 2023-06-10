import torch as to

test_prompt = """### Instruction:

Summarize the input text in 3 different lengths:

- 50 words
- 5 words
- a single word

### Input:

In the heart of a bustling metropolis, a captivating tapestry of urban
life unfolds before your eyes. Majestic skyscrapers reach toward the heavens,
their sleek facades reflecting the vibrant energy that permeates the streets
below. The rhythm of life pulses through every corner, as a kaleidoscope of
diverse cultures intertwines, creating a vibrant mosaic of human experiences.
Sidewalks teem with a symphony of sounds, from the chatter of passersby to
the enticing melodies spilling out from cozy cafes and lively music venues.
The air is alive with the aromas of world cuisines, wafting from bustling food
carts and acclaimed restaurants, offering a tantalizing culinary journey for
adventurous palates. Vibrant street art adorns the walls, transforming once
ordinary spaces into canvases of creativity, telling stories of passion,
resilience, and the human spirit. Parks and green spaces provide a sanctuary
amidst the urban bustle, inviting city dwellers to reconnect with nature and
find solace in the midst of the concrete jungle. Technology hums in the
background, driving innovation and connecting people in ways never before
imagined. Markets bustle with activity, where the vibrant colors of fresh
produce, the intricate craftsmanship of local artisans, and the friendly
banter of vendors create a sensory feast for visitors and locals alike. In
this vibrant tapestry of urban life, dreams take flight, connections are
forged, and the city's pulse beats with endless possibilities. It is a place
where the past and the future converge, inviting all who enter to embrace the
energy, seize the opportunities, and create their own unique story within the
vibrant embrace of the metropolis.

### Response:"""

def generate( device, model, tokenizer, input_text ):
    used_input_text = test_prompt
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
            max_new_tokens = 200,
            top_p = 0.95,
            top_k = 50,
            temperature = 0.8,
            repetition_penalty=1.02
        )

    output_text = tokenizer.decode( [ el.item() for el in generated_ids[ 0 ] ] )
    return output_text.replace( "<s>", "" ).replace( "</s>", "" ) \
        .replace( used_input_text, "" ).strip()