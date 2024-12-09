from copy import deepcopy
import json
import random
from typing import Dict, List
import numpy as np
import pandas as pd

ranking_prompts = ['GPT-P1', 'GPT-P2',
                   'GPT-P3', 'GPT-P4',
                   'GPT-P4A', 'GPT-P5']
generating_prompts = ['GPT-P1N', 'GPT-P2N',
                      'GPT-P3N', 'GPT-P4N',
                      'GPT-P4NA', 'GPT-P5N',
                      'GPT-P6N', 'GPT-P6']
sentiment_to_number = {
    "Positive": 0,
    "Negative": 1,
    "Neutral": 2
}


def extract_sentiment(sentiment: str):
    sentiment = sentiment.lower()
    if "positive" in sentiment:
        return "Positive"
    elif "neutral" in sentiment:
        return "Neutral"
    elif "negative" in sentiment:
        return "Negative"
    elif "sell" in sentiment:
        return "Negative"
    elif "buy" in sentiment:
        return "Positive"
    else:
        return sentiment


def sentiment_to_numeric(sentiment: str):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return 1.0
    elif sentiment == 'negative':
        return -1.0
    elif sentiment == 'neutral':
        return 0.0
    else:
        return np.nan


def sample_random_examples(dataset: pd.DataFrame, ticker: str, example_index: int, n_shots: int, random_state: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    ticker_df = dataset.loc[(dataset['ticker'] == ticker) & ~(
        dataset.index.isin([example_index])), :]
    sample_df = ticker_df.sample(
        n=n_shots,
        random_state=random_state
    )
    return sample_df


def generate_messages(sample_contents: List, ticker: str, prompt_type: str, prompt_details: Dict):
    sample_messages = []

    for sample_content in sample_contents:
        messages = deepcopy(prompt_details['messages'])

        if prompt_type in ['GPT-P4A', 'GPT-P4AN']:
            for message in messages:
                message['content'] = message['content'].replace('{ticker}', ticker).replace('{headline}', sample_content).replace(
                    '{article}', sample_content)
        elif prompt_type in ["GPT-P3", "GPT-P3N"]:
            for message in messages:
                message['content'] = message['content'].replace(
                    '{headline}', sample_content)
        else:
            for message in messages:
                message['content'] = message['content'].replace(
                    '{ticker}', ticker).replace('{headline}', sample_content)

        sample_messages.append(messages)

    return sample_messages


def format_prompt(ticker, contents, sample_answers, prompt_type, model_type, prompts_file='prompts.json'):
    """
    Get sentiment using a specified prompt type.

    Parameters:
    - ticker: The forex ticker.
    - contents: The article or headline.
    - prompt_type: The type of prompt to use (e.g., "GPT-P4A", "GPT-P1N").
    - prompts_file: Path to the JSON file containing the prompts.

    Returns:
    - List of message for LLMs.
    """
    # Get the question and examples
    actual_content = contents[0]
    sample_contents = contents[1:]

    # Load prompts from the JSON file
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    # Get the selected prompt details
    prompt_details = prompts[prompt_type]
    index_messages = generate_messages(
        [actual_content], ticker, prompt_type, prompt_details)[0]
    sample_messages = generate_messages(
        sample_contents, ticker, prompt_type, prompt_details)

    index_messages[-1]['content'] = "{message}\n Answer:".format(
        message=index_messages[-1]['content'])

    merged_sample_message = ""
    if len(sample_messages) > 0:
        merged_sample_messages = []
        for i, sample_message in enumerate(sample_messages):
            sample_contents = [message['content']
                               for message in sample_message]
            sample_contents[-1] = "{message}\n Answer: {answer}.".format(
                message=sample_contents[-1],
                answer=sample_answers[i]
            )
            sample_value = "\n ".join(sample_contents)
            merged_sample_messages.append(sample_value)
        merged_sample_message = "\n \n ".join(merged_sample_messages)

    if model_type == 'BERT':
        message_contents = [message['content'] for message in index_messages]
        index_messages = "\n ".join(message_contents)
        if merged_sample_message != "":
            index_messages = '{example}\n \n {message}'.format(
                example=merged_sample_message,
                message=index_messages
            )

        index_messages = index_messages.replace("\n", "\\n")
        index_messages = [index_messages]
    else:
        if merged_sample_message != "":
            # Append the reference example to system prompt
            index_messages[0]['content'] = 'By referring to the examples provided below:\n {example}\n \n {message}'.format(
                example=merged_sample_message,
                message=index_messages[0]['content']
            )

    return index_messages
