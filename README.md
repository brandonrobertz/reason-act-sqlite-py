# Reason-Act SQLite Demo

![Reason and Act Flow Chart](https://github.com/brandonrobertz/reason-act-sqlite-py/blob/main/reason-act-llm.png?raw=true)

This is a demonstration of how to use [reason and act][react-paper] with [llama.cpp][llama-git] and a [LLM][xwin] to pose plain english queries to a SQLite database using one of two strategies:

1. Actions that mimic interaction with a frontend like [Datasette][datasette-git]. Actions: list tables, list table columns, facet, filter
2. Let the LLM use SQLite queries directly. Actions: list tables, list table schema, execute sql

The things you'll need to do are:

1. Provide a SQLite database (named `example.db` or you need to change the name in the Python files)
2. Change the prompts in both Python scripts (the `prompt` string inside the `execute` functions) to be specific to your data and problems. You'll also want to date the `DATA_HELP` table and column descriptions in `run-sql-queries.py`.
3. Download a GGUF model for use. The default is to look for [dolphin-2.2.1-mistral-7b.Q5_K_M.gguf][dolphin-2.2.1-mistral-7b] in the current dir. If you want to use a different model, edit the script you're running.

There are some dependencies for this project that you need, first. You can install with using pip:

```
pip install -r requirements.txt
```

Once you have everything installed and configured, you can kick off a session by coming up with a question and asking it on the command line:

```
python run_interface.py "What kind of data do I have available?"
python llm_sql_queries.py "What are some interesting records in the database?"
```

The model output will be printed to stdout.

[react-paper]: https://blog.research.google/2022/11/react-synergizing-reasoning-and-acting.html?m=1
    "ReAct: Synergizing Reasoning and Acting in Language Models"

[llama-git]: https://github.com/ggerganov/llama.cpp
    "Port of Facebook's LLaMA model in C/C++"

[xwin]: https://huggingface.co/TheBloke/Xwin-LM-13B-V0.1-GGUF
    "Xwin-LM-13B-V0.1-GGUF on Huggingface courtesy of TheBloke"

[datasette-git]: https://github.com/simonw/datasette
    "An open source multi-tool for exploring and publishing data"

[llama-cpp-py]: https://github.com/abetlen/llama-cpp-python
    "Python bindings for llama.cpp"

[sqlite-utils]: https://github.com/simonw/sqlite-utils
    "Python CLI utility and library for manipulating SQLite databases"

[dolphin-2.2.1-mistral-7b]: https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/tree/main
    "Dolphin 2.2.1 Mistral 7B GGUF on HuggingFace"
