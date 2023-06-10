import torch as to

# test_prompt = """### Instruction:

# Summarize the input text in 3 different lengths:

# - 50 words
# - 5 words
# - a single word

# ### Input:

# Tokyo's architecture is a mesmerizing blend of tradition and innovation,
# showcasing the city's unique character. Walking through the bustling streets,
# one is captivated by the seamless juxtaposition of ancient temples and
# ultramodern skyscrapers. The iconic Tokyo Tower stands tall, reminiscent of
# the Eiffel Tower, offering panoramic views of the city's architectural marvels.
# The imperial palace, surrounded by lush gardens, presents a stunning display
# of classical Japanese design, with its intricate wooden structures and tranquil
# moats. Tokyo's skyline is dominated by sleek glass towers that house
# futuristic offices and luxury apartments, reflecting the city's status as a
# global economic powerhouse. The vibrant Shibuya crossing is not just a
# bustling intersection but a testament to Tokyo's architectural innovation, with
# its illuminated billboards and towering facades. Traditional wooden machiya
# townhouses, with their distinctive latticed windows and tiled roofs, can still
# be found tucked away in the historic neighborhoods, providing a glimpse into
# Tokyo's past. Architectural wonders like the Meiji Shrine, with its majestic
# torii gates and serene forest surroundings, offer a serene escape from the urban
# chaos. Tokyo's architectural landscape constantly evolves, with cutting-edge
# designs such as the iconic Tokyo Skytree, a towering communication tower that
# boasts a futuristic aesthetic and serves as a symbol of Japan's technological
# advancements. From ancient temples to modern skyscrapers, Tokyo's architecture
# is a testament to the city's rich heritage and its unwavering embrace of
# progress.

# ### Response:

# # 50 words

# Tokyo's architecture seamlessly blends tradition and innovation, showcasing
#     ancient temples, ultramodern skyscrapers, iconic landmarks like Tokyo
#     Tower and Meiji Shrine, and vibrant neighborhoods like Shibuya. The city's
#     architectural marvels reflect its rich heritage and commitment to
#     progress, making it a captivating global hub of design.

# # 4 words

# Traditional and modern Tokyo.

# # a single word

# Tokyo

# ### Instruction:

# Summarize the input text in 3 different lengths:

# - 50 words
# - 5 words
# - a single word

# ### Input:

# In the heart of a bustling metropolis, a captivating tapestry of urban
# life unfolds before your eyes. Majestic skyscrapers reach toward the heavens,
# their sleek facades reflecting the vibrant energy that permeates the streets
# below. The rhythm of life pulses through every corner, as a kaleidoscope of
# diverse cultures intertwines, creating a vibrant mosaic of human experiences.
# Sidewalks teem with a symphony of sounds, from the chatter of passersby to
# the enticing melodies spilling out from cozy cafes and lively music venues.
# The air is alive with the aromas of world cuisines, wafting from bustling food
# carts and acclaimed restaurants, offering a tantalizing culinary journey for
# adventurous palates. Vibrant street art adorns the walls, transforming once
# ordinary spaces into canvases of creativity, telling stories of passion,
# resilience, and the human spirit. Parks and green spaces provide a sanctuary
# amidst the urban bustle, inviting city dwellers to reconnect with nature and
# find solace in the midst of the concrete jungle. Technology hums in the
# background, driving innovation and connecting people in ways never before
# imagined. Markets bustle with activity, where the vibrant colors of fresh
# produce, the intricate craftsmanship of local artisans, and the friendly
# banter of vendors create a sensory feast for visitors and locals alike. In
# this vibrant tapestry of urban life, dreams take flight, connections are
# forged, and the city's pulse beats with endless possibilities. It is a place
# where the past and the future converge, inviting all who enter to embrace the
# energy, seize the opportunities, and create their own unique story within the
# vibrant embrace of the metropolis.

# ### Response:"""

# test_prompt = """Q1:

# \"\"\"
# Summarize the below quoted text in 3 different lengths:

# - 50 words
# - 5 words
# - a single word

# '''
# Tokyo's architecture is a mesmerizing blend of tradition and innovation,
# showcasing the city's unique character. Walking through the bustling streets,
# one is captivated by the seamless juxtaposition of ancient temples and
# ultramodern skyscrapers. The iconic Tokyo Tower stands tall, reminiscent of
# the Eiffel Tower, offering panoramic views of the city's architectural marvels.
# The imperial palace, surrounded by lush gardens, presents a stunning display
# of classical Japanese design, with its intricate wooden structures and tranquil
# moats. Tokyo's skyline is dominated by sleek glass towers that house
# futuristic offices and luxury apartments, reflecting the city's status as a
# global economic powerhouse. The vibrant Shibuya crossing is not just a
# bustling intersection but a testament to Tokyo's architectural innovation, with
# its illuminated billboards and towering facades. Traditional wooden machiya
# townhouses, with their distinctive latticed windows and tiled roofs, can still
# be found tucked away in the historic neighborhoods, providing a glimpse into
# Tokyo's past. Architectural wonders like the Meiji Shrine, with its majestic
# torii gates and serene forest surroundings, offer a serene escape from the urban
# chaos. Tokyo's architectural landscape constantly evolves, with cutting-edge
# designs such as the iconic Tokyo Skytree, a towering communication tower that
# boasts a futuristic aesthetic and serves as a symbol of Japan's technological
# advancements. From ancient temples to modern skyscrapers, Tokyo's architecture
# is a testament to the city's rich heritage and its unwavering embrace of
# progress.
# '''
# \"\"\"

# A1:

# \"\"\"
# # 50 words

# Tokyo's architecture seamlessly blends tradition and innovation, showcasing
#     ancient temples, ultramodern skyscrapers, iconic landmarks like Tokyo
#     Tower and Meiji Shrine, and vibrant neighborhoods like Shibuya. The city's
#     architectural marvels reflect its rich heritage and commitment to
#     progress, making it a captivating global hub of design.

# # 5 words

# Traditional and modern Tokyo.

# # a single word

# Tokyo
# \"\"\"

# Q2 ( new topic ):

# \"\"\"
# Summarize the below quoted text in different lengths:

# - 50 words
# - 5 words
# - a single word

# '''
# In the heart of a bustling metropolis, a captivating tapestry of urban
# life unfolds before your eyes. Majestic skyscrapers reach toward the heavens,
# their sleek facades reflecting the vibrant energy that permeates the streets
# below. The rhythm of life pulses through every corner, as a kaleidoscope of
# diverse cultures intertwines, creating a vibrant mosaic of human experiences.
# Sidewalks teem with a symphony of sounds, from the chatter of passersby to
# the enticing melodies spilling out from cozy cafes and lively music venues.
# The air is alive with the aromas of world cuisines, wafting from bustling food
# carts and acclaimed restaurants, offering a tantalizing culinary journey for
# adventurous palates. Vibrant street art adorns the walls, transforming once
# ordinary spaces into canvases of creativity, telling stories of passion,
# resilience, and the human spirit. Parks and green spaces provide a sanctuary
# amidst the urban bustle, inviting city dwellers to reconnect with nature and
# find solace in the midst of the concrete jungle. Technology hums in the
# background, driving innovation and connecting people in ways never before
# imagined. Markets bustle with activity, where the vibrant colors of fresh
# produce, the intricate craftsmanship of local artisans, and the friendly
# banter of vendors create a sensory feast for visitors and locals alike. In
# this vibrant tapestry of urban life, dreams take flight, connections are
# forged, and the city's pulse beats with endless possibilities. It is a place
# where the past and the future converge, inviting all who enter to embrace the
# energy, seize the opportunities, and create their own unique story within the
# vibrant embrace of the metropolis.
# '''
# \"\"\"

# A2:

# \"\"\""""

# test_prompt = """Ron: If I have an apple and a pear, how many fruits do I have?
# Mary: One apple plus one pear makes two pieces of fruit.
# Ron: If I have a peach and a strawberry, how many fruits do I have?
# Mary: One peach plus"""

# test_prompt = """Ron: My buddy Donald is going to visit me tomorrow. My buddy's name is Donald.
# Mark: My buddy Stephen is going to visit me tomorrow. My buddy's name is Stephen.

# _Stephen calls._

# Stephen: Hey, Mark, """

# test_prompt = """Alice was so tired when she got back home so she went"""

test_prompt = """'I need to write a summary for a text about animals, mom.
How can I do this?', Tom asked his mother.
"""

def generate( device, model, tokenizer, input_text ):
    used_input_text = test_prompt
    input_ids = tokenizer.encode( used_input_text, return_tensors = "pt" ) \
        .to( device )

    with to.no_grad():
        generated_ids = model.generate(
            input_ids,
            # use_cache=True, do_sample=True,
            # eos_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
            # min_length = 500,
            # max_length = 512,
            max_new_tokens = 1000,
            # top_p = 0.95,
            # top_k = 50,
            # temperature = 0.3,
            # repetition_penalty=1.02,
            # num_beams = 1
        )

    output_text = tokenizer.decode(
        [ el.item() for el in generated_ids[ 0 ] ], skip_special_tokens = True
    )
    return output_text.replace( "<s>", "" ).replace( "</s>", "" ) \
        .replace( used_input_text, "" ).strip()