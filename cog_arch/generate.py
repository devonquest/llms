import time as tm
import re

def count_tokens( message ):
    words = re.split( r'\s+', message )
    num_tokens = len( words ) / 0.75

    return int( num_tokens )

def measure( inferred_text, before, after ):
    num_tokens = count_tokens( inferred_text )

    return num_tokens, num_tokens / ( after - before )

def generate_once( prompt, max_new_tokens, generate, tokenizer ):
    before = tm.time()
    response = generate(
        prompt,
        temperature=0.8, top_p=0.95, top_k=50, max_new_tokens=max_new_tokens,
        use_cache=True, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.02
    )

    after = tm.time()
    text = response[ 0 ][ "generated_text" ]
    text = text.replace( prompt, "" ).replace( "\"\"\"", "" ).strip()

    return text, *measure( text, before, after )

def generate_with_predicate(prompt, predicate, generate, tokenizer):
    generation_attempts = 0
    response, *_ = generate_once(prompt, 3, generate, tokenizer)
    
    if not predicate(response):
        generation_attempts += 1
        if generation_attempts == 10:
            print("Generation attempts exhausted, exiting...")
            exit(1)
        print(f"\nInvalid response:\n\n{response}")
        return generate_with_predicate(prompt, predicate)

    return response

# <!-- TODO: Write outline for following idea -->
# <!-- - llm receives prompt with new text input -->
# <!-- - llm creates 4 different summarizations -->
# <!--    - 1 word -->
# <!--    - 3 words -->
# <!--    - 6 words -->
# <!--    - 50 words -->
# <!-- - these summarizations are stored in persistent storage in a fixed 4
#         layer hierarchy -->
# <!--    - exact number may be iterated on, point is non-growing for constant
#             lookup -->
# <!-- - a sufficiently brief conversational history summarization is
#         maintained -->
# <!--    - summarization is created once conversational history size is deemed
#             large enogh by LLM -->
# <!--    - with ( included in ) each new text input prompt the LLM judges
#             whether that summarization is still roughly in line with the context
#             of the new input ( enables among other things tangents ) -->
# <!--    - if it's deemed not in line, a new one is created and pushed into
#             the summarization list ( head most recent ) -->
# <!--    - once an array of these exists, each new text input prompt will be
#             followed up by an intermediate inference pass that determines the
#             most appropriate summarization from the array -->
# <!--        - if none is found appropriate a new one will be created as usual
#                 -->
# <!--    - each newly created summarization will be linked to a set or subset
#             of text input summarizations on the lower ( more detailed )
#             summarization layers using a dict ( one per layer ) -->
# <!--        - if a dict already has the key ( can occur in some necessary
#                 performance related techniques later described ), a new,
#                 slightly different summarization and a new corresponding
#                 subset will be created and linked -->
# <!--    - the purpose of this is key to a cognition mechanism with constant
#             time memory lookup -->
# <!--        - new summarizations will ( relatively ) infrequently be created ->
# <!--        - no need to iterate all text input summarizations in a given
#                 layer in order to gather relevant inference details ->
# <!--        - summarization is non-changing during a given conversation and
#                 therefore provides deterministic, layer-subset limited lookup ->
# <!--        - rich distant knowledge base associations are still included by
#                 LLM pickig various different one word summarizations in the 
#                 first inference pass ->
# <!--        - once history summarization array exceeds context window,
#                 the array is split in half and the new parts are linked
#                 at the cutting point ->
# <!--            - for the case in which the LLM chooses the oldest ( tail end )
#                     summary of the curent array ->
# <!--                - a subsequent inference pass is performed on the tail
#                         linked ( from previous split ) array ->
# <!--                - this kind of array jumping can possibly occur in streaks
#                         , which may be reasonably pointed towards in a prompt ->
# <!--                - in order to make very old lookup efficient, the jumping
#                         gap will be increased based on an exponential
#                         function ->
# <!--                - the previous makes practically infinitely old memory
#                         lookup possible ->
# <!--                - once an inference chooses a summarization in an older
#                         array as appropriate, that array will be inserted
#                         before the head as a memory reinforcement ->
# <!-- - each input text summarization nests the more detailed ones -->
# <!-- - abstraction and pruning is performed only on the top level ones -->
# <!--    - once multiple one word summarizations occur -->
# <!--        - merge their 3 word summarizations under one  -->
# <!--        - if 3 words aren't enough to semi-guarantee ( ignoring rare
#                 edge cases as a reasonable tradeoff ), that number can be
#                 increased  -->
# <!-- - once one of the lower level summarization layers or subsets therein
#         exceed the context window -->
# <!--    - its least relevant summarizations will be extracted into a new subset
#             -->
# <!--    - if the current / original layer / subset is not yet linked to the
#             current history summarization, it will be  -->
# <!--    - a new, slightly different ( as influenced by the new subset )
#             history summarization will be created for and linked to the subset
#             -->
# <!--    - this mechanism along with the maintenance of the flat, linked
#             structure of history summarizations provides constant time memory
#             lookup where each subset has relevance to a period in time
#             -->
# <!-- - using the above described mechanisms each relevant memory retrieval
#         phase for text input processing
#         only ever involves 4 inference passes plus a couple infrequent ones
#         in cases of detected history summarization change, subset
#         extraction / split and appropriate history summarization lookup -->
# <!--    - on each integration, the max available tokens ( as determined
#             by context window ) will be distributed based on relevance to the
#             one word summarizations and similar distributions are done in the
#             three deeper layers
#             -->
# <!--    - it must be made clear to the LLM that it may be necessary, even usual
#             to distribute large amounts of tokens to individual summarizations
#             due to strong topic foci -->
# <!--    - it must also be made clear that some summarization inclusions may be
#             prematurely stopped at early layers -->
# <!--    - this leads to a mix of summarizations in very different detail levels
#             based on relevance which provides rich associations across the
#             whole knowledge base-->
# <!--    - this leads to a mix of summarizations in very different detail levels
#             based on relevance which provides rich associations across the
#             whole knowledge base -->
# <!--    - a relevant completion can then be provided based on the assembled
#             context -->