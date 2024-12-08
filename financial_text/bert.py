import copy
import re
import string
from typing import Dict
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True).eval().cuda()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@torch.no_grad()
def generating_predict(
        model: Dict,
        text: str,
        max_tokens: int = 16):
    input_ids = model["tokenizer"](
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.cuda()

    # Generate text using the pre-trained model
    prediction = model["model"].generate(
        input_ids,
        num_beams=4,
        do_sample=False,
        use_cache=None,
        max_new_tokens=max_tokens,
        min_new_tokens=2,
        temperature=1.0,
        eos_token_id=[model["tokenizer"](eos, add_special_tokens=False).input_ids[1] for eos in [
            ".\\", "\\.", "\\,", "\\;"]]
    )
    prediction = prediction[0, input_ids.size(1):]
    prediction = model["tokenizer"].decode(prediction)
    prediction = prediction.replace('\\n', "\n")
    prediction = prediction.strip().strip('\\')
    prediction = prediction.strip().split('\n')[0].strip().strip('\\').strip()
    # prediction = normalize_answer(prediction)

    return prediction


def ranking_predict(model, answers, message, verbose=False):
    logps = []
    for option in answers:
        inputs = "{message} {answer}".format(message, option)
        score_length = len(inputs.split())

        if verbose:
            print(inputs)

        logp = model["model"].score(
            inputs,
            score_length,
            model["tokenizer"],
            "cuda",
            8
        )
        logps.append(logp)

    prediction = logps.index(max(logps))
    return prediction - 1
