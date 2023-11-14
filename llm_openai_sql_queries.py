import json
import os
import re
import sys
import sqlite3
import time

import openai
import sqlite_utils

from llm_sql_queries import (
    DB_PATH, load_db,
    tables, schema, help, sql_query
)


# Larger context sizes will reduce quality, but some models
# support large contexts better than others.
#CONTEXT_SIZE=2048
CONTEXT_SIZE=2048*2
# how many tokens to allow the model to output in a sigle go w/o stopping
MAX_TOKENS=400


def execute(model_path, outfile=None, debug=True, return_dict=None,
            prompt=None, n_gpu_layers=0, temp=None, top_p=None):
    assert prompt, "You didn't supply a prompt"
    db = load_db(DB_PATH)
    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    action_fns = {
        "tables":  tables,
        # "columns": columns,
        "schema": schema,
        "help": help,
        "sql-query": sql_query,
    }

    if debug:
        print(prompt)

    total_tokens = 0
    done = False
    while not done:
        model_name = model_path.split(":", 1)[1]
        print("Running OpenAI model", model_name)
        print("Prompt", prompt)
        sys.exit(1)
        model_kwargs = dict(
            # model="gpt-4",
            model=model_name,
            # string / array / null
            # Up to 4 sequences where the API will stop generating
            # further tokens. The returned text will not contain the
            # stop sequence.
            stop=[
                "Question:", "Observation:",
                "<|im_end|>", "<|im_start|>user",
            ],
            stream=True,
            messages=prompt,
        )
        # Open AI recommends not using BOTH temperature and top-p
        if temp is not None:
            model_kwargs["temperature"] = temp
        elif top_p is not None:
            model_kwargs["top_p"] = top_p

        try:
            stream = openai.ChatCompletion.create(
                **model_kwargs
            )
        except openai.error.RateLimitError:
            print("Cooling down...")
            time.sleep(30)

        with open("debug-openai.log", "a") as f:
            f.write(json.dumps(prompt))
            f.write('\n')

        response = ""
        for i, item in enumerate(stream):
            # {
            #   "choices": [
            #       {
            #           "delta": {
            #               "role": "assistant"
            #               # OR, once started a role
            #               "content": "\n\n"
            #           },
            #           "finish_reason": null | "stop",
            #           "index": 0
            #       }
            #   ],
            #   "created": 1677825464,
            #   "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            #   "model": "gpt-3.5-turbo-0301",
            #   "object": "chat.completion.chunk"
            # }
            if i > MAX_TOKENS:
                break
            choice = item['choices'][0]
            print(i, json.dumps(choice), end="\t\t\t\t\t\r")

            # if it gives a non-assistant role, end
            role = choice["delta"].get("role")
            if role and role != "assistant":
                break
            # if it wants to stop (or hits a stopword) let it
            if choice.get("finish_reason") == "stop":
                break

            total_tokens += 1
            if total_tokens > CONTEXT_SIZE:
                done = True
                break

            # otherwise assume we have another token
            token = choice["delta"]["content"]
            response += token

            with open("debug-openai.log", "a") as f:
                f.write(json.dumps(item))
                f.write('\n')

        # Update the prompt
        prompt.append({"role": "assistant", "content": response})

        if debug:
            # print(f"{prompt}\n")
            print(response)

        with open("debug-openai.log", "a") as f:
            f.write(json.dumps(prompt))
            f.write('\n')

        if outfile:
            with open(outfile, "w") as f:
                f.write(json.dumps(prompt, indent=2))

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
            prompt.append({
                "role": "user",
                "content": f"Observation: That's an invalid action. Valid actions: {action_names}"
            })

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
            prompt.append({
                "role": "user",
                "content": f"Observation: {observation_text}"
            })

        elif final_answer:
            if return_dict is not None:
                return_dict["final_answer"] = final_answer
                return_dict["trace"] = prompt
            return final_answer, prompt

        # TODO: truncate the prompt if its grown too long
        # using tiktoken and some keep_n value of context

    if return_dict is not None:
        return_dict["final_answer"] = None
        return_dict["trace"] = prompt

    return None, prompt


if __name__ == "__main__":
    question = sys.argv[1]
    outfile = None
    try:
        outfile = sys.argv[2]
    except IndexError:
        pass
    answer, trace = execute("openai:gpt-4", question, outfile=outfile)
    print("Full trace")
    print("="*72)
    print(trace)
    print("-"*72)
    print("Final Answer:", answer)
