import json
import os
import re
import sys
import sqlite3

from llama_cpp import Llama

from actions import (
    DB_PATH, load_db,
    tables, schema, help, sql_query
)


# Larger context sizes will reduce quality, but some models
# support large contexts better than others.
#CONTEXT_SIZE=2048
CONTEXT_SIZE=2048*2
# how many tokens to allow the model to output in a sigle go w/o stopping
MAX_TOKENS=400


# Utils n stuff
def load_model(model_path, n_gpu_layers=0, n_threads=os.cpu_count() - 1,
               n_ctx=CONTEXT_SIZE, temp=None, top_p=None):
    # for LLaMA2 70B models add kwarg: n_gqa=8 (NOTE: not required for GGUF models)
    print("Loading model", model_path)
    print("CTX:", n_ctx, "GPU layers:", n_gpu_layers, "CPU threads:", n_threads)
    print("Temperature:", temp, "Top-p Sampling:", top_p)
    kwargs = dict(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False
    )
    if temp is not None:
        kwargs["temp"] = temp
    if top_p is not None:
        kwargs["top_p"] = top_p
    llm = Llama(**kwargs)
    return llm


def execute(model_path, outfile=None, debug=True, return_dict=None,
            prompt=None, n_gpu_layers=0, temp=None, top_p=None):
    llm = load_model(model_path, n_gpu_layers=n_gpu_layers, temp=temp,
                     top_p=top_p)
    db = load_db(DB_PATH)
    action_fns = {
        "tables":  tables,
        "schema": schema,
        "help": help,
        "sql-query": sql_query,
    }
    action_names_text = ", ".join(list(action_fns.keys()))
    prompt_is_chatml = "<|im_start|>" in prompt
    if debug:
        print(prompt)

    n_sequential_whitespace = 0
    n_thoughts_seen = 0
    done = False
    while not done:
        stream = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            stop=["Question:", "Observation:", "<|im_end|>", "<|im_start|>user"],
            stream=True,
            echo=True
        )
        response = ""
        for i, token in enumerate(stream):
            choice = token['choices'][0]
            print(i, choice, end="\t\t\t\t\t\r")
            token = choice["text"]
            response += token
            if token in ["", "\n"]:
                n_sequential_whitespace += 1
            else:
                n_sequential_whitespace = 0
            # detect repeating loop
            if response.count("Thought: ") > 4:
                done = True
                break
            if n_sequential_whitespace > 20:
                done = True
                break

            with open("debug.log", "a") as f:
                f.write(json.dumps(token))
                f.write('\n')

        if prompt_is_chatml and not response.strip().endswith("<|im_end|>"):
            response = f"{response.strip()}\n<|im_end|>\n"

        # Update the prompt
        prompt = f"{prompt}{response}".strip()

        if debug:
            print(response)

        if outfile:
            with open(outfile, "w") as f:
                f.write(prompt)

        if done:
            break

        try:
            action = re.findall(r"Action: (.*)", response, re.M)[0]
        except IndexError:
            action = None

        try:
            final_answer = re.findall(r'Final Answer: (.*)', response, re.M|re.S)[0]
        except IndexError:
            final_answer = None

        if action and action not in action_fns:
            action_names = ", ".join(list(action_fns.keys()))
            if prompt_is_chatml:
                prompt += f"""
<|im_start|>user
Observation: That's an invalid action. Valid actions: {action_names}
<|im_end|>
<|im_start|>assistant
Thought: """
            else:
                prompt += f"""Observation: That's an invalid action. Valid actions: {action_names}
Thought: """

        elif action:
            # NOTE: we could change 1 for the number of args of selected action
            actionInputs = re.findall(
                r'Action Input (\d): ```([^`]+)```', response, re.M|re.S
            )
            args = [
                inp[1]
                for inp in actionInputs
            ]
            action_fn = action_fns[action]
            observation_text = ""
            try:
                print("Running action", action_fn, end="... \t")
                result = action_fn(db, *args)
                print("Done!", end="\r")
                result_text = json.dumps(result)
                observation_text = f"```{result_text}```"
            except TypeError as e:
                if "positional argument" not in str(e):
                    raise e
                # trim off the name of of the action from msg like:
                #   hi() takes 1 positional argument but 2 were given
                # and turn it into:
                #   The action hi takes 1 Action Input but 2 were given
                args_err_msg = str(e).split(" ", 1)[1].replace(
                    "positional argument", "Action Input"
                ).replace(
                    "positional arguments", "Action Inputs"
                ).split(": '", 1)[0]
                observation_text = f"The action {action} {args_err_msg}"
            if prompt_is_chatml:
                prompt += f"""
<|im_start|>user
Observation: {observation_text}
<|im_end|>
<|im_start|>assistant
Thought: """
            else:
                prompt += f"""
Observation: {observation_text}
Thought: """

        elif final_answer:
            if return_dict is not None:
                return_dict["final_answer"] = final_answer.replace(
                    "<|im_end|>", ""
                ).strip()
                return_dict["trace"] = prompt
            return final_answer, prompt

        # TODO: truncate the prompt if its grown too long
        # using tiktoken and some keep_n value of context

    if return_dict is not None:
        return_dict["final_answer"] = None
        return_dict["trace"] = prompt
    return None, prompt
