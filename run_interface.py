import json
import os
import re
import sys
import sqlite3

from llama_cpp import Llama
import sqlite_utils


DB_PATH = "example.db"
MODEL_PATH = "dolphin-2.2.1-mistral-7b.Q5_K_M.gguf"
# columns to not ever use or show
IGNORED_COLUMNS = ["rowid", "created_at", "_meta_score"]
TABLE_FTS = {
    "users": ["creatorDescription"],
    "jobs": ["title", "description"],
    "experiences": ["projectName", "experienceDescription", "jobRole"],
    "games": ["name", "description"],
    "game_passes": ["name", "description"],
}
# search-tables: useful for getting a list of tables that support full-text search. input 1: table name.
# search: a full-text search engine. useful to find records with descriptions containing some text. input 1: table name, input 2: a search query.
ACTIONS = """
tables: useful for getting the names of tables available. no input.
columns: useful for looking all of the columns for a given table. input 1: table name.
facets: useful for looking at the unique values and counts for a given column. input 1: table name, input 2: column name.
filter: useful for getting the first row where the column matches a given value. input 1: table name, input 2: column name, input 3: a value to filter on.
"""


def load_db(path):
    assert os.path.exists(path), f"Database doesn't exist: {path}"
    db = sqlite_utils.Database(path)
    for table, fields in TABLE_FTS.items():
        try:
            db[table].enable_fts(fields, create_triggers=True)
        except sqlite3.OperationalError:
            pass
    return db


def is_array_field(db, table_name, column):
    # test if it's an array type, first result will always start with '['
    rows = db.query(f"""
        select {column} as value
        from {table_name}
        where {column} is not null and {column} != ""
        limit 1
    """)
    for row in rows:
        if isinstance(row["value"], int):
            return False
        return row["value"].startswith("[")


def clean_truncate(results, n=3):
    return [
        {k: v for k, v in r.items() if k not in IGNORED_COLUMNS}
        for r in results[:n]
    ]


## ACTIONS
def tables(db):
    return [
        name
        for name in db.table_names()
        if "_fts" not in name
    ]


def columns(db, table_name):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Invalid table. Valid tables are: {table_names}"
    return [
        c.name
        for c in db[table_name].columns
        if c.name not in IGNORED_COLUMNS
    ]


def facets(db, table_name, column):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Invalid table. Valid tables are: {table_names}"
    column_names = columns(db, table_name)
    if column not in column_names:
        return f"Invalid column. Valid columns are: {column_names}"
    if is_array_field(db, table_name, column):
        results = db.query(f"""
            SELECT value, count(*) AS count
            FROM (SELECT j.value AS value
                  FROM {table_name}
                  CROSS JOIN json_each({table_name}.{column}) AS j)
            GROUP BY value
            ORDER BY count DESC
            LIMIT 5;
        """)
    else:
        # if it's not an array type
        results = db.query(f"""
            SELECT {column} AS value, count({column}) AS count
            FROM {table_name}
            GROUP BY {column}
            ORDER BY count DESC
            LIMIT 5
        """)
    return [
        [r["value"], r["count"]]
        for r in results
    ]


def filter(db, table_name, column, value):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Invalid table. Valid tables are: {table_names}"
    column_names = columns(db, table_name)
    if column not in column_names:
        return f"Invalid column. Valid columns are: {column_names}"
    if is_array_field(db, table_name, column):
        results = list(db.query(f"""
            SELECT *
            FROM {table_name}
            WHERE EXISTS (SELECT 1 FROM json_each({column}) WHERE value = '{value}')
        """))
    else:
        results = list(db[table_name].rows_where(f"{column} = ?", [value]))
    return clean_truncate(results, n=1)


def search(db, table_name, query):
    results = list(db[table_name].search(query))
    return clean_truncate(results)


# Utils n stuff
def load_model(model_path):
    return Llama(model_path=model_path, n_ctx=2048)


def execute(llm, question):
    action_fns = {
        "tables":  tables,
        "columns": columns,
        "facets": facets,
        "filter": filter,
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
Action Input 2: the second input to the action, if more than one.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here's an example:

Question: What information do I have about users?
Thought: I should check to see if I have any users tables.
Action: tables
Observation: ["jobs", "users", "games", "game_stats", "game_passes"]
Thought: I should inspect the columns on the users table.
Action: columns
Action Input 1: "users"
Observation: ["creatorUserId", "isPublic", "isContactAllowed", "creatorDescription", "isOpenToWork", "interestDescription", "jobTypes", "skillTypes"]
Thought: I have all of the information I need.
Final Answer: I have the following fields about users: ["creatorUserId", "isPublic", "isContactAllowed", "creatorDescription", "isOpenToWork", "interestDescription", "jobTypes", "skillTypes"]

Here's another example:

Question: What job types are being offered?
Thought: I should look at which tables I have available.
Action: tables
Observation: ["jobs", "users", "games", "game_stats", "game_passes"]
Thought: I should see which columns are available for jobs.
Action: columns
Action Input 1: "jobs"
Observation: ["jobType", "paymentTypes", "isVerified", "paymentAmountType", "paymentTypes", "skillTypes"]
Thought: I should look at the unique values the jobType column has on the jobs table.
Action: facets
Action Input 1: "jobs"
Action Input 2: "jobType"
Observation: [["PartTime", 6753]]
Thought: It looks like there is only one job type so I have the final answer.
Final Answer: There is only one type of job being offered: PartTime (6,754 jobs).

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
            final_answer = re.findall(r'Final Answer: (.*)', response, re.M)[0]
        except IndexError:
            final_answer = None
        # print("** final answer:", final_answer)

        if action and action not in action_fns:
            prompt = output["choices"][0]["text"].strip()
            action_names = ", ".join(list(action_fns.keys()))
            prompt += f"\nObservation: That's an invalid action. Valid actions: {action_names}\n"

        elif action:
            # force to have quotes
            actionInputs = re.findall(
                r'Action Input (\d): "(.*)"', response, re.M
            )
            args = [
                inp[1]
                for inp in actionInputs
            ]
            action_fn = action_fns[action]
            # print("** action_fn:", action_fn, "args", args)
            try:
                observation = action_fn(db, *args)
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
                observation = f"The action {action} {args_err_msg}"
            # print("** observation:", observation)
            observation_text = json.dumps(observation)
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
