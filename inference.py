from os import system
from contextlib import ExitStack

from llama_cpp import Llama

llm = Llama("./zephyr3B.gguf", n_gpu_layers=9999)


def generate():
    with ExitStack() as s:
        sys_prompt, user_prompts = "", []

        with open("system/default.txt") as f:
            sys_prompt = f.read()

        for n in range(2):
            f = open(f"./prompts/user/breakdown_task/{n}.txt")
            s.enter_context(f)

            prompt = f.read()
            user_prompts.append(prompt)

        output = ""

        for p, i in enumerate(user_prompts):
            if i > 0:
                p += output

            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": p}
                ],
                max_tokens=150
            )

            msg = "\n-- Sys prompt\n\n"
            msg += f"{ sys_prompt }\n\n"
            msg += "-- User prompt\n\n"
            msg += f"{ p }\n\n"
            msg += "-- Response\n\n"
            msg += output["choices"][0]["message"]["content"]
            msg += "\n"

            print(f"{msg}\n\n")
            input("Press enter to continue.\n")


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
