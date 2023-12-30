from os import system
from contextlib import ExitStack
import re

from llama_cpp import Llama

llm = Llama("./zephyr3B.gguf", n_gpu_layers=9999)


def generate():
    with ExitStack() as s:
        sys_prompt, user_prompts = "", []

        with open("./prompts/system/default.txt") as f:
            sys_prompt = f.read()

        for n in range(2):
            f = open(f"./prompts/user/breakdown_task/{n}.txt")
            s.enter_context(f)

            prompt = f.read()
            user_prompts.append(prompt)

        output = ""

        for i, p in enumerate(user_prompts):
            if i > 0:
                lines, trimmed = output.split("\n"), ""

                for l in lines:
                    if bool(re.match("^\d.*$")):
                        trimmed += f"\n{l}"
                trimmed = trimmed.strip()

                # TODO: Remove ( debug )
                print(trimmed)
                input("Continue.")

                p += trimmed

            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": p}
                ],
                max_tokens=150
            )["choices"][0]["message"]["content"]

            msg = "\n-- Sys prompt\n\n"
            msg += f"{ sys_prompt }\n\n"
            msg += "-- User prompt\n\n"
            msg += f"{ p }\n\n"
            msg += "-- Response\n\n"
            msg += output
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
