#!/usr/bin/env python
from datetime import datetime
import multiprocessing
import json
import os
import re
import sys
import time

import numpy as np
import pymeteor.pymeteor as pymeteor
from nltk.corpus import stopwords
from nltk import download
import spacy
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from metrics import get_keyword_matches
from llm_sql_queries import execute
from llm_openai_sql_queries import execute as execute_openai


USE_EXAMPLE_INJECTION = False
# HACK: globals
nlp = None
stop_words = None


def load_yml_file(filename):
    with open(filename, "r") as f:
        return load(f, Loader=Loader)


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


def best_matching_injectable(question, injectables):
    best = [0.0, injectables[0]["prompt"]]
    q1 = " ".join(preprocess(question))
    question_vec = nlp(q1)
    for injectable in injectables:
        q2 = " ".join(preprocess(injectable["question"]))
        inj_question_vec = nlp(q2)
        sim = inj_question_vec.similarity(question_vec)
        print(sim, "Q:", q1, "Q2:", q2)
        if sim > best[0]:
            best = [sim, injectable["prompt"]]
    return best[1]


def maybe_inject_prompts(prompt_data, question, injectables=None):
    if not USE_EXAMPLE_INJECTION:
        return prompt_data

    if not injectables:
        return prompt_data

    if not nlp:
        return prompt_data

    similar_injectable = best_matching_injectable(question, injectables)

    # first: truncate the examples by looking for the inject_before: True
    # on the prompt items
    truncate_at = None
    for i, item in enumerate(prompt_data):
        should_inject = item.pop("inject_before", False)
        if should_inject:
            truncate_at = i
            break

    if truncate_at is None:
        return prompt_data

    # This also cuts off the final part, we need to fix that
    truncated_prompt_data = prompt_data[:i] + similar_injectable
    # append the question now
    truncated_prompt_data.append(prompt_data[-1])
    return truncated_prompt_data


def prompt_data_to_openai(prompt_data, question, injectables=None):
    prompt_completed = maybe_inject_prompts(prompt_data, question, injectables=injectables)
    prompt_completed[-1]["content"] = prompt_completed[-1]["content"].format(question=question)
    return prompt_completed


def prompt_data_to_raw(prompt_data, question, injectables=None):
    prompt_completed = maybe_inject_prompts(prompt_data, question, injectables=injectables)
    prompt_raw = ""
    for item in prompt_completed:
        line = item["content"].format(question=question)
        prompt_raw += line
        prompt_raw += "\n"
        if "Final Answer:" in line:
            prompt_raw += "\n"
    return prompt_raw.strip()


def prompt_data_to_chatml(prompt_data, question, injectables=None):
    prompt_completed = maybe_inject_prompts(prompt_data, question, injectables=injectables)
    prompt_raw = ""
    last_item = len(prompt_completed) - 1
    for i, item in enumerate(prompt_completed):
        line = item["content"].format(question=question).strip()
        if item["role"] == "system":
            prompt_raw += "<|im_start|>system\n"
            prompt_raw += f"{line}\n<|im_end|>\n"

        if item["role"] == "assistant":
            prompt_raw += "<|im_start|>system name=example_assistant\n"
            prompt_raw += f"{line}\n<|im_end|>\n"
            if "Final Answer: " in line:
                prompt_raw += "\n"

        if item["role"] == "user" and i != (last_item):
            prompt_raw += "<|im_start|>system name=example_user\n"
            prompt_raw += f"{line}\n<|im_end|>\n"

        # the final one is the question with the lead out for completion
        if item["role"] == "user" and i == (last_item):
            prompt_raw += "<|im_start|>user\n"
            prompt_raw += f"{line}\n<|im_end|>\n"
            prompt_raw += "<|im_start|>assistant\n"
            prompt_raw += "Thought: "

    return prompt_raw.strip()


def get_model_name(model_file):
    model_name=re.sub('[^A-Za-z0-9\-_]+', "_", os.path.basename(model_file))
    return model_name


def get_tracefile(model_file):
    model_name = get_model_name(model_file)
    now=datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    tracefile = f"./traces/experiment_{model_name}_{now}.log"
    return tracefile


def run_llm(*args, timeout=30*60, **kwargs):
    # shared dict for transferring results back from the proc
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    kwargs["return_dict"] = return_dict

    execute_fn = execute
    if args[0].startswith("openai:"):
        execute_fn = execute_openai

    p = multiprocessing.Process(
        target=execute_fn, name="LLM",
        args=args, kwargs=kwargs
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise Exception(f"Timed out after {timeout}s")

    if not return_dict:
        print("Blank return_dict. Likely an error!")

    return return_dict.get("final_answer"), return_dict.get("trace")


def save_experiment_data(experiment_output, experiment_data):
    print("Writing experiment data to", experiment_output)
    with open(experiment_output, "w") as f:
        f.write(json.dumps(experiment_data, indent=2))


def run_experiment(
    model_path, prompt_data, qa, experiment_output,
    cooldown=None, n_tries=10, n_gpu_layers=0,
    temp=None, top_p=None,
    injectables=None, timeout=30*60
):
    experiment_data = {
        "question_results": [],
        "model_name": get_model_name(model_path),
        "model_path": model_path,
        "prompt": prompt_data,
    }
    for q_data in qa:
        q_result = q_data.copy()
        print()
        print("="*72)
        print("Beginning with question:", q_result["question"])
        q_result["scores"] = []
        q_result["tracefiles"] = []
        q_result["errors"] = []
        q_result["keyword_matches"] = []
        q_result["answers"] = []
        for i in range(n_tries):
            print()
            print("-" * 72)
            print(f"Attempt: {i}")

            question = q_result["question"]
            prompt = None
            if experiment_prompt == "raw":
                prompt = prompt_data_to_raw(prompt_data, question, injectables=injectables)
            elif experiment_prompt == "chatml":
                prompt = prompt_data_to_chatml(prompt_data, question, injectables=injectables)
            elif experiment_prompt == "openai":
                prompt = prompt_data_to_openai(prompt_data, question, injectables=injectables)

            tracefile = get_tracefile(model_path)
            correct_keywords = q_result["correct_keywords"]

            print("Writing to:", tracefile)
            print("Question:", question)
            answer = None
            error = None
            try:
                answer, trace = run_llm(
                    model_path, outfile=tracefile,
                    debug=False, prompt=prompt,
                    n_gpu_layers=n_gpu_layers,
                    timeout=timeout, temp=temp,
                    top_p=top_p
                )
            except Exception as e:
                print(f"ERROR: {e}")
                error = f"{e}"

            print("Answer:", answer)
            reference = q_result["correct_answer"]
            candidate = answer or ""

            meteor_score = pymeteor.meteor(reference, candidate, print_details=True)
            print("Score:", meteor_score)
            keyword_matches = get_keyword_matches(candidate, correct_keywords)
            print("Keyword Matches:", keyword_matches)

            q_result["scores"].append(meteor_score)
            q_result["tracefiles"].append(tracefile)
            q_result["errors"].append(error)
            q_result["keyword_matches"].append(keyword_matches)
            q_result["answers"].append(answer)

            save_experiment_data(experiment_output, experiment_data)

            if cooldown:
                print(f"Cooling down for {cooldown}s...")
                time.sleep(cooldown)

        print("Appending experiment")
        experiment_data["question_results"].append(q_result)
        print(len(experiment_data["question_results"]),
              "experiments have been completed")

        save_experiment_data(experiment_output, experiment_data)

    return experiment_data


if __name__ == "__main__":
    try:
        experiment_plan_file = sys.argv[1]
    except IndexError:
        print("USAGE: benchmark.py experiment_plan_file")
        print("Where experiment_plan_file points to a yaml file describing the experiments to be performed.")
        sys.exit(1)

    print("Loading experiment plan file", experiment_plan_file)
    experiment_plan = load_yml_file(experiment_plan_file)

    today=datetime.now().strftime("%Y-%m-%d")
    exp_name = experiment_plan["EXPERIMENT_NAME"]
    timeout = experiment_plan.get("TIMEOUT")

    prompt_data = experiment_plan["PROMPT_DATA"]
    injectables = experiment_plan.get("AVAILABLE_INJECT_PROMPTS")

    if USE_EXAMPLE_INJECTION:
        print("Loading NLP models")
        nlp = spacy.load("en_core_web_lg")
        download('stopwords')  # Download stopwords list.
        stop_words = stopwords.words('english')

    for model_data in experiment_plan["MODELS"]:
        experiment_model = model_data["path"]
        experiment_prompt = model_data["prompt_type"]

        model_name = get_model_name(experiment_model)
        experiment_output = f"./experiments/{exp_name}_{today}_{model_name}.json"

        experiment_data = run_experiment(
            experiment_model,
            prompt_data,
            experiment_plan["QA"],
            experiment_output,
            cooldown=model_data.get("cooldown") or experiment_plan.get("COOLDOWN"),
            n_tries=experiment_plan["N_TRIES"],
            n_gpu_layers=model_data.get("n_gpu_layers", 0),
            temp=experiment_plan.get("temp"),
            top_p=experiment_plan.get("top_p"),
            injectables=injectables,
            timeout=model_data.get("timeout", timeout)
        )
        save_experiment_data(experiment_output, experiment_data)
