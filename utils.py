import json
import re

from jsonschema import validate
from jsonschema import ValidationError

from schema import SCHEMA


def has_pii(text):
    """
    Detect simple email or phone numbers.
    """

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    phone_pattern = (
        r"\b\d{10}\b|"
        r"\b\d{3}[-.]\s]\d{3}[-.]\s]\d{4}\b"
    )

    return bool(
        re.search(email_pattern, text)
        or re.search(phone_pattern, text)
    )


def fallback():
    """
    Return fallback JSON.
    """

    return {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None,
    }


def validate_response(response_text):
    """
    Validate JSON returned by the LLM.
    """

    try:

        data = json.loads(response_text.strip())

    except json.JSONDecodeError as e:

        print("JSON Error:", e)

        return fallback()

    try:

        validate(instance=data, schema=SCHEMA)

    except ValidationError as e:

        print("Validation Error:")

        print(e.message)

        return fallback()

    return data
