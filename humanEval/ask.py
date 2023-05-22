# TO AID WITH REPRODUCIBILITY...
# ENVIRONMENT ============================= #
# Python                             3.11.3 #
#                                           #
# LIBRARIES ------------------------------- #
# sleepyask                           6.2.0 #
# pandas                              2.0.1 #
# ========================================= #

import os
from dotenv import load_dotenv

import pandas
from sleepyask.chat import Sleepyask

load_dotenv()  # take environment variables from .env.

def create_prompt(row):
    PREFIX = "Write a program based on the docstring below. Do not explain your code. Denote your code by tagging it within '''python ''':\n\n\n"
    PROMPT = row['prompt']

    return {"text": f"{PREFIX}\n{PROMPT}", "id": row["task_id"]}


# ChatGPT configs
IN_PATH = 'in/ground.jsonl'
OUT_PATH = "out/1-base.jsonl"

TIMEOUT = 1000
RETRY_TIME = 5
RATE_LIMIT = 1000
API_KEY = os.getenv('OPENAI_API_KEY')

CONFIGS = {
    "model": "gpt-3.5-turbo",
    "n": 50
}

QUESTION_LIST = pandas.read_json(IN_PATH, lines=True)
QUESTION_LIST = QUESTION_LIST.apply(lambda row: create_prompt(row), axis=1).to_list()

sleepyask = Sleepyask(configs=CONFIGS, 
                      rate_limit=RATE_LIMIT,
                      api_key=API_KEY, 
                      timeout=TIMEOUT, 
                      out_path=OUT_PATH, 
                      verbose=True,
                      retry_time=RETRY_TIME)
sleepyask.start(question_list=QUESTION_LIST)
