# CSE476_FinalProject_Group17

## Running the project
open a terminal in the src folder, and run:

    python3 generate_answers.py

## Uploading your code
Make sure to push your code to your own branch rather than pushing to main. Then open a pull request on the github website.

## .env file
Create a file called '.env' in the '/src' folder. Inside it put the line

    OPENAI_API_KEY="YOUR_KEY"

But replace YOUR_KEY with your actual api key


## pip
Run the following command to make sure you have the dependencies

    pip install requests python-dotenv

## VPN
Make sure you are using the Cisco VPN.

## The 8 distinct inference-time algorithms or techniques used

- Decomposition
- Task Classification
- Synthetic Context Generation
- Context Injection
- Self Consistency
- CoT
- Self-Refine
- LLM-as-a-judge

## Total # of LLM calls
Total LLM Calls: 3 + [#ofsubtasks*6]
With 1 subtask: 9 LLM calls
With 2 subtasks: 15 LLM calls
With 3 subtasks: 21 LLM calls