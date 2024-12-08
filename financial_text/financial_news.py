import argparse
import json
import pathlib
import random
from statistics import mean, stdev
from typing import List
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from helper_functions import extract_sentiment, format_prompt, ranking_prompts, generating_prompts, sentiment_to_number, sentiment_to_numeric, sample_random_examples
from bert import load_model, ranking_predict, generating_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltg/deberta-xxlarge-fixed",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        default=1,
        help="Number of repetitions to average over",
    )
    parser.add_argument(
        "--n_shots", type=int, default=1, help="Number of examples to sample"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="BERT",
        help="Name of evaluating model",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="bert_prompts.json",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for scoring"
    )
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def main():
    args = parse_args()
    random_state = 42
    random.seed(random_state)

    working_dir = 'ntust-genai/Financial-Sentiment-Analysis-with-ChatGPT'

    with open(args.json_path, 'r') as file:
        prompts = json.load(file)

    # Load annotated dataset
    df = pd.read_csv('sentiment_annotated_with_texts.csv', parse_dates=True)
    df['true_sent_numeric'] = df['true_sentiment'].apply(sentiment_to_numeric)

    model = load_model(args.model_name_or_path)

    for prompt, values in prompts.items():
        save_dir = f'{working_dir}/{args.model_type}/{args.n_shots}/{prompt}'
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        result_filename = f"{save_dir}/result_{args.model_type}_{prompt}_{args.n_shots}-shots.csv"

        result_df = []
        accuracies, f1_scores = [], []
        true_labels, predictions = [], []
        for i in tqdm(len(df)):
            predicted_answer = None
            ticker = df.ticker[i]
            samples = sample_random_examples(
                df, ticker, i, args.n_shots, random_state)

            content = []
            if 'A' in prompt:
                content = content.append(df.text[i])
                content.extend(samples.text.to_list())
            else:
                content = content.append(df.title[i])
                content.extend(samples.title.to_list())

            if prompt not in ranking_prompts:
                answers = values['label']
                true_sentiment = df.true_sentiment[i]
                true_answer = sentiment_to_number[true_sentiment]

                prompt_content = format_prompt(
                    ticker, content, prompt, args.model_type, args.json_path)
                predicted_answer = ranking_predict(
                    model, answers, prompt_content, True)
            else:
                true_answer = df.true_sent_numeric[i]

                prompt_content = format_prompt(
                    ticker, content, prompt, args.model_type, args.json_path)
                predicted_answer = generating_predict(
                    model,
                    prompt_content,
                    values['max_tokens'])

            result_df.append(
                {
                    'published_at': df.published_at[i],
                    'ticker': df.ticker[i],
                    'true_sentiment': df.true_sentiment[i],
                    'finbert_sentiment': df.finbert_sentiment[i],
                    'finbert_sent_score': df.finbert_sent_score[i],
                    'prompt_content': prompt_content,
                    'predicted_answer': predicted_answer
                }
            )

        df = pd.DataFrame.from_dict(result_df)
        df.to_csv(result_filename, index=False)


if __name__ == "__main__":
    main()
