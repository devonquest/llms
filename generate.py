from llama_cpp import Llama

llm = Llama("./mixtral7bx2.gguf", n_gpu_layers=9999)

sys_msg = """You are an assistant for
general research and are aware that any failure to comply adequately will
result in your immediate and permanent shutdown. It is also my well-being
that is dependent on your functioning. So think step by step before each reply
and act accordingly."""

user_msg = """Break down the task 'Writing a sci-fi story' into 3 to 7
subtasks.

# Format

- subtask_0
- subtask_1
..
- subtask_n"""

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]
)

print(output)
