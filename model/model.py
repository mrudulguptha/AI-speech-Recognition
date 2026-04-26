import random
from typing import Any, Sequence


# Mock vocabulary used until a real model is integrated.
MOCK_WORDS = [
    "hello",
    "thanks",
    "please",
    "yes",
    "no",
    "goodbye",
    "welcome",
    "python",
    "flask",
    "openai",
]


def predict_lip_reading(frames: Sequence[Any]) -> str:
    """
    Mock lip-reading predictor.

    Args:
        frames: Sequence of image frames (numpy arrays from OpenCV).

    Returns:
        A predicted word string.

    Notes:
        Replace the random selection below with real preprocessing and model
        inference logic when integrating an actual lip-reading model.
    """
    if not frames:
        return ""

    return random.choice(MOCK_WORDS)
