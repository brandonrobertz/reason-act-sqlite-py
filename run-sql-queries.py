import json
import os
import re
import sys
import sqlite3

from llama_cpp import Llama
import sqlite_utils


DB_PATH = "example.db"

# Fast for testing and is very good. It doesn't seem to use group by or more
# advanced features though
# MODEL_PATH = "../models/xwin-lm-13b-v0.1.Q6_K.gguf"

# Mistral is great, esp for a 7B model. I can run this on my 8G 1080 GPU
# with a N_GPU_LAYERS=-1 setting below
MODEL_PATH = "../models/mistral-7b-v0.1.Q5_K_M.gguf"

# # Looks promising but I need to convert my prompts to ChatML
# MODEL_PATH = "../models/dolphin-2.1-mistral-7b.Q5_K_S.gguf"

# # Higher quality much slower
# MODEL_PATH = "../models/xwin-lm-70b-v0.1.Q2_K.gguf"

# # Trying out SQLcoder, it can do advanced SQL. Result: dumpster fire
# MODEL_PATH = "../models/sqlcoder.Q5_K_M.gguf"

# # Does weird things with SQL
# MODEL_PATH = "../models/vicuna-33b.Q6_K.gguf"

# # Fastest, makes lots of mistakes
# MODEL_PATH = "../models/llama-2-7b-chat.Q5_K_S.gguf"

# If you have a GPU, tune this so that it's as large as possible w/o
# getting an out of memory error. For my 8G vram GPU I can use the
# following layer counts for the xwin 7b Q5_K_M
#
# Model Size    Layers
# 70b           10
# 13b           22
#  7b           -1
N_GPU_LAYERS=-1

# Larger context sizes will reduce quality, but some models
# support large contexts better than others.
CONTEXT_SIZE=2048

ACTIONS = """
tables: useful for getting the names of tables available. no input.
schema: useful for looking at the schema of a database. input 1: table name.
help: get helpful information describing a table or a table's column. useful for understanding the relationship between tables and what columns mean. input 1: table name. (optional) input 2: column name.
sql-query: useful for analyzing data and getting the top 5 results of a query. input 1: a valid sqlite sql query.
"""

DATA_HELP = {
    "table name": {
        None: "enter a table description here",
        "column name": "a description of the column",
    },
}

# tables to omit from commands
IGNORED_TABLES = [
    "game_stats",
    "ar_internal_metadata",
    "schema_migrations",
    "filing_folders",
]
IGNORED_COLUMNS = []


def load_db(path):
    assert os.path.exists(path), f"Database doesn't exist: {path}"
    db = sqlite_utils.Database(path)
    return db


def clean_truncate(results, n=3):
    return [
        {k: v for k, v in r.items()}
        for r in results[:n]
    ]


## ACTIONS
def tables(db):
    return [
        name
        for name in db.table_names()
        # game stats confuses the model
        if "_fts" not in name and name not in IGNORED_TABLES
    ]


def schema(db, table_name):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Invalid table. Valid tables are: {table_names}"
    return db[table_name].schema


def help(db, *args):
    table_name = args[0]
    column = None
    if len(args) == 2:
        column = args[1]
    if table_name not in DATA_HELP:
        available_tables = tables(db)
        return f"The table {table_name} doesn't exist. Valid tables: ```{available_tables}```"
    if column not in DATA_HELP[table_name]:
        available_columns = [
            c.name
            for c in db[table_name].columns
            if c.name not in IGNORED_COLUMNS
        ]
        return f"The column {column} isn't in the {table_name} table. Valid columns: ```{available_columns}```"
    return DATA_HELP[table_name][column]


def sql_query(db, query):
    # if "select *" in query.lower():
    #     return "Error: Select specific columns, not *"
    try:
        results = list(db.query(query))
    except sqlite3.OperationalError as e:
        return f"Your query has an error: {e}"
    return clean_truncate(results, n=5)


# Utils n stuff
def load_model(model_path):
    # for LLaMA2 70B models add kwarg: n_gqa=8 (NOTE: not required for GGUF models)
    return Llama(model_path=model_path, n_ctx=CONTEXT_SIZE,
                 n_gpu_layers=N_GPU_LAYERS, n_threads=os.cpu_count() - 1,
                 verbose=False)


def execute(llm, question):
    action_fns = {
        "tables":  tables,
        "schema": schema,
        "help": help,
        "sql-query": sql_query,
    }
    action_names_text = ", ".join(list(action_fns.keys()))
    prompt = f"""
Answer the following questions as best you can. You have access to the following tools:

{ACTIONS.strip()}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{action_names_text}]
Action Input 1: the first input to the action.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here's an example:

Question: What information do I have about users?
Thought: I should check to see if I have any users tables.
Action: tables
Observation: ```["jobs", "users", "games", "game_passes"]```
Thought: I should inspect the columns on the users table.
Action: schema
Action Input 1: ```users```
Observation: ```CREATE TABLE [users] ( [creatorUserId] INTEGER PRIMARY KEY, [isContactAllowed] INTEGER, [creatorDescription] TEXT, [isOpenToWork] INTEGER, [interestDescription] TEXT, [linkTypes] TEXT, [preferredContactLinkType] TEXT, [socialLinks] TEXT, [jobTypes] TEXT, [skillTypes] TEXT, [requiresAction] TEXT )```
Thought: I should see what ways the users table can be helpful by checking the help.
Action: help
Action Input 1: ```users```
Observation: users are individuals who are seeking work, have worked or are looking to hire people to work on games. sometimes they mention demographic information about themselves in their description.
Thought: I have all of the information I need.
Final Answer: I have the following fields about users: "creatorUserId", "isContactAllowed", "creatorDescription", "isOpenToWork", "interestDescription", "jobTypes", "skillTypes", and "requiresAction".

Here's another example:

Question: How many jobs have been offered?
Thought: I should look at which tables I have available.
Action: tables
Observation: ```["jobs", "users", "games", "game_passes"]```
Thought: I should count the rows of the jobs table using SQLite SQL.
Action: sql-query
Action Input 1: ```select count(*) from jobs;```
Observation: ```[{{'count(*)': 6753}}]```
Thought: This query has given me the count of jobs in the table. I have a final answer.
Final Answer: There have been 6,753 total jobs offered according to the database.

You can run any SQLite query to answer aggregate questions:

Question: What are the most common payment amounts that have been offered for jobs?
Thought: I should look at which tables I have available.
Action: tables
Observation: ```["jobs", "users", "games", "game_passes"]```
Thought: I should use a SQL group by query to see the top jobs.paymentAmount values.
Action: sql-query
Action Input 1: ```select paymentAmount, count(paymentAmount) as n from jobs group by paymentAmount limit 3;```
Observation: ```[{{"paymentAmount": 0.0, "n": 713}}, {{"paymentAmount": 1.0, "n": 157}}, {{"paymentAmount": 2.0, "n": 22}}]```
Thought: This query has given me the count of the top payment amounts in the
jobs table. I have a final answer.
Final Answer: The top three payment amounts offered for jobs are 0.0 (713 jobs), 1.0 (157), and 2.0 (22 jobs).

You can get helpful information about table and columns by using the help action:

Question: What kind of information shows up in the user description?
Thought: I should read the help for the description column in the users table.
Action: help
Action Input 1: ```users```
Action Input 2: ```creatorDescription```
Observation: The users table's creatorDescription column is a free-text field the user has supplied, describing themselves, their interests and work preferences.
Thought: I have some information about what appears in user descriptions.
Final Answer: Users sometimes put their interests, work preferences and demographic information in their description.

Begin!

Question: {question.strip()}
Thought:""".strip()
    print(prompt)

    # allow the LLM to try 15 goes to get to an answer
    attempts = 0
    while attempts < 15:
        attempts += 1
        # print(f"** running attempt: {attempts}")

        output = llm(
            prompt,
            max_tokens=256,
            stop=["Question:", "Observation:"],
            echo=True
        )
        # print("** output:", output)

        # get the response minus our prompt (just the new stuff)
        response = output["choices"][0]["text"].replace(prompt, "").strip()
        #print(f"** response:\n{response}")
        print(f"\n{response}")

        try:
            action = re.findall(r"Action: (.*)", response, re.M)[0]
        except IndexError:
            action = None
        # print("** action:", action)

        try:
            final_answer = re.findall(r'Final Answer: (.*)', response, re.M|re.S)[0]
        except IndexError:
            final_answer = None
        # print("** final answer:", final_answer)

        if action and action not in action_fns:
            prompt = output["choices"][0]["text"].strip()
            action_names = ", ".join(list(action_fns.keys()))
            prompt += f"\nObservation: That's an invalid action. Valid actions: {action_names}\n"

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
            # print("** action_fn:", action_fn, "args", args)
            observation_text = ""
            try:
                result = action_fn(db, *args)
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
            # print("** observation:", observation)
            print(f"Observation: {observation_text}\n")
            prompt = output["choices"][0]["text"].strip()
            prompt += f"\nObservation: {observation_text}\n"

        elif final_answer:
            trace = output["choices"][0]["text"]
            return final_answer, trace

        # TODO: truncate the prompt if its grown too long
        # using tiktoken and some keep_n value of context

    return None, output["choices"][0]["text"]


if __name__ == "__main__":
    question = sys.argv[1]
    db = load_db(DB_PATH)
    llm = load_model(MODEL_PATH)
    answer, trace = execute(llm, question)
