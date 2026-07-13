"""
JSON Schema for validation
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {
            "type": "string"
        },
        "confidence_level": {
            "type": "string"
        },
        "top_reason": {
            "type": "string"
        },
        "second_reason": {
            "type": "string"
        },
        "next_step": {
            "type": "string"
        }
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ]
}
