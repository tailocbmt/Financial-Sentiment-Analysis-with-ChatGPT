import json
import time
import logging
from typing import Dict, List
import numpy as np
import openai


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gpt_prediction(model_type: str, messages: List, prompt_details: Dict, max_retries=5):
    """
    Get sentiment using a specified prompt type.

    Parameters:
    - model_type: GPT Model (gpt-3.5-turbo/gpt-4o).
    - messages: Message to .
    - prompt_details: The prompt other value.
    - max_retries: Maximum number of retries in case of an error.

    Returns:
    - Sentiment.
    """
    time.sleep(0.1)

    for retry in range(max_retries):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=model_type,
                messages=messages,
                temperature=prompt_details.get('temperature', 0),
                max_tokens=prompt_details.get('max_tokens', 16),
                top_p=prompt_details.get('top_p', 1),

            )
            end_time = time.time()
            response_time = end_time - start_time

            return response.choices[0].message.content
        except Exception as e:
            logger.error(
                f"Error while processing '{messages}' (Retry {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                # Sleep for a while before retrying
                time.sleep(0.1)
            else:
                # If all retries fail, return NaN values
                return np.nan
