#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

from llm import run_agent

import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_PATH = Path("./data/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("./data/cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding='latin-1') as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def save_answers(path: Path, answers: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)


def build_answers(
    questions: List[Dict[str, Any]], start_index: int = 1
) -> List[Dict[str, str]]:

    answers = []

    if (OUTPUT_PATH.exists()):
        with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
            answers = json.load(fp)

    for idx, question in enumerate(questions[start_index - 1 :], start=start_index):
        real_answer = run_agent(question["input"], "")
        answers.append({"output": real_answer})
        save_answers(OUTPUT_PATH, answers)
        print(f"Finished question {idx}")
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions, 1)

    with OUTPUT_PATH.open("r", encoding='utf-8') as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()


