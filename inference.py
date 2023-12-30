from os import system
from contextlib import ExitStack

from llama_cpp import Llama

llm = Llama("./zephyr3B.gguf", n_gpu_layers=9999)


def generate():
    with ExitStack() as s:
        prompts = []

        for n in ("sys_prompt", "user_prompt"):
            f = open(f"{n}.txt")
            s.enter_context(f)

            prompt = f.read()
            prompts.append(prompt)

        sys_prompt, user_prompt = prompts
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=70
        )

        msg = output["choices"][0]["message"]["content"]
        msg += "\n\n"
        print(msg)


def run_loop():
    menu = """Menu
    
- generate ( g )
- pull ( p )
- end ( e )\n\n"""

    system("clear")
    selection = input(menu)
    system("clear")

    if selection not in ("g", "p", "e"):
        print("Invalid selection. Try again.")
    elif selection == "g":
        generate()
    elif selection == "p":
        system("git pull")
        print("\n\n")

    if selection != "e":
        input("Press enter to continue.")
        run_loop()


run_loop()
