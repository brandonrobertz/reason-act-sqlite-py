#!/usr/bin/env python
import json
import os
import re
import sys

import pandas as pd
import pymeteor.pymeteor as pymeteor
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from metrics import get_keyword_matches


def load_yml_file(filename):
    with open(filename, "r") as f:
        return load(f, Loader=Loader)


def final_answer_from_trace_or_result(tracepath, result=None):
    final_answer = ""
    if not os.path.exists(tracepath):
        return result or ""
    with open(tracepath, "r") as f:
        lines = f.readlines()
        i = 0
        while "Final Answer: " in "\n".join(lines):
            try:
                line = lines.pop()
            except IndexError:
                return final_answer
            if not line.startswith("Final Answer:"):
                continue
            final_answer += line
            if line.startswith("<|im_end|>"):
                break
            if line.startswith("Thought: "):
                break
            if line.startswith("Question: "):
                break
            i += 1
    return final_answer.replace("<|im_end|>", "")


if __name__ == "__main__":
    benchmark_plan_file = None
    results_outfile = None
    try:
        benchmark_plan_file = sys.argv[1]
        results_outfile = sys.argv[2]
    except IndexError:
        print("USAGE: rescore.py benchmark-file.yml results_output.csv")
        sys.exit(1)

    plan = load_yml_file(benchmark_plan_file)
    question_keywords = {
        qa["question"]: qa["correct_keywords"]
        for qa in plan["QA"]
    }
    question_answers = {
        qa["question"]: qa["correct_answer"]
        for qa in plan["QA"]
    }

    results = [
        ["Experiment", "Model", "Task", "Keyword(s)", "METEOR", "Match Texts"],
    ]
    for basedir, subdirs, filenames in os.walk("./experiments/"):
        for filename in filenames:
            print("Loading experiment file", filename)
            with open(os.path.join(basedir, filename), "r") as f:
                try:
                    experiment = json.load(f)
                except json.decoder.JSONDecodeError:
                    continue
            try:
                experiment_name = re.findall(r"^(.+)_2023-\d\d-\d\d_.*", filename)[0]
            except Exception:
                continue
            model_name = experiment["model_name"]
            print("=" * 72)
            print("Experiment:", experiment_name, "Model name:", model_name)

            # TODO: replace these with cli arg filters
            if "ROBLOX" not in experiment_name or "ROBLOX" not in filename:
                continue
            # if "OPENAI" not in experiment_name:
            #     continue

            for q_n, result in enumerate(experiment["question_results"]):
                question = result["question"]
                correct_keywords = question_keywords[question]
                correct_answer = question_answers[question]

                keyword_matches_all = []
                meteor_score_all = []
                match_texts_all = []
                for index in range(len(result["scores"])):
                    print("-" * 72)
                    print(f"Q_{q_n}_{index}")

                    error = result["errors"][index]

                    tracefile = result["tracefiles"][index]
                    print("tracefile", tracefile)
                    if not os.path.exists(tracefile):
                        tracefile = "missing"

                    final_answer = result["answers"][index] or ""
                    # final_answer = final_answer_from_trace_or_result(
                    #     tracefile, result=result["answers"][index]
                    # )
                    print("Final answer:", final_answer)

                    keyword_score, match_texts = get_keyword_matches(
                        final_answer, correct_keywords, return_texts=True
                    )
                    match_texts_all.append(match_texts)
                    print("Calculating METEOR on:", final_answer)
                    meteor_score = pymeteor.meteor(
                        correct_answer, final_answer
                    )
                    exp_result = [
                        experiment_name,
                        model_name,
                        f"Q_{q_n}_{index}",
                        keyword_score,
                        meteor_score,
                        match_texts,
                    ]
                    print(exp_result)
                    results.append(exp_result)
                    meteor_score_all.append(meteor_score)
                    print("METEOR score:", meteor_score)
                    keyword_matches_all.append(keyword_score)
                    print("Keyword score:", keyword_score)

                print("-" * 72)
                print(f"Q_{q_n} Aggregates")
                exp_result = [
                    experiment_name,
                    model_name,
                    f"Q_{q_n}_Avg",
                    sum(keyword_matches_all) / len(keyword_matches_all),
                    sum(meteor_score_all) / len(meteor_score_all),
                    match_texts_all,
                ]
                results.append(exp_result)
                print(exp_result)

                exp_result = [
                    experiment_name,
                    model_name,
                    f"Q_{q_n}_Max",
                    max(keyword_matches_all),
                    max(meteor_score_all),
                    match_texts_all,
                ]
                results.append(exp_result)
                print(exp_result)

    results_df = pd.DataFrame(results[1:], columns=results[0])
    with open(results_outfile, "w") as f:
        f.write(results_df.to_csv())

    # import IPython
    # IPython.embed()
    # import time
    # time.sleep(2)
