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
    PREFIX = "Solve the following logic question. End your response with the phrase 'The answer is...' where you respond whether you think the answer is 'Yes', 'No'."
    FACTS = '\n'.join(row['facts'])
    QUESTION = row['question']

    return {"text": f"{PREFIX}\n{QUESTION}Facts:\n{FACTS}", "id": row["qid"]}


# ChatGPT configs
IN_PATH = 'in/ground.json'
OUT_PATH = "out/1-baseline-temp-0.jsonl"


TIMEOUT = 100
RETRY_TIME = 5
RATE_LIMIT = 300
API_KEY = os.getenv('OPENAI_API_KEY')

CONFIGS = {
    "model": "gpt-3.5-turbo",
    "n": 50
}

QUESTION_LIST = pandas.read_json(IN_PATH)
QUESTION_LIST = QUESTION_LIST.apply(
    lambda row: create_prompt(row), axis=1).to_list()

sleepyask = Sleepyask(configs=CONFIGS, 
                      rate_limit=RATE_LIMIT,
                      api_key=API_KEY, 
                      timeout=TIMEOUT, 
                      out_path=OUT_PATH, 
                      verbose=False,
                      retry_time=RETRY_TIME)
sleepyask.start(question_list=QUESTION_LIST[0:501])
