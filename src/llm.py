# %% Minimal setup
# If needed (uncomment in a notebook):
# !pip install requests python-dotenv

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507") 

def agent_loop (question: str) -> str:
    return API_KEY
    #return ("ANSWER TO: " + question.encode('utf-8').decode('utf-8'))[:100] + API_KEY

def testin():
    return "HI"