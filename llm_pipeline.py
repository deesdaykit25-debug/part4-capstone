"""
Part 4: LLM-Powered Model Explanation Pipeline
Track C - Model Prediction Explanation Pipeline

This script integrates the Part 3 classification model with OpenRouter LLM
to generate structured JSON explanations of insurance charge predictions.
"""

import os
import json
import requests
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, USER_TEMPLATE
from utils import has_pii, validate_response, fallback
from schema import SCHEMA

# Load environment variables from .env file
load_dotenv()

PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUTS_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUTS_FOLDER.mkdir(exist_ok=True)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("LLM_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "LLM_API_KEY not found in environment variables. "
        "Please create a .env file with: LLM_API_KEY=your_key_here"
    )


def call_llm(prompt_text, temperature=0):
    """
    Call OpenRouter LLM with structured prompt.
    
    Args:
        prompt_text (str): The user prompt to send to the LLM
        temperature (float): Controls randomness (0=deterministic, 0.7=creative)
    
    Returns:
        str: LLM response text, or None on error
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": temperature,
        "max_tokens": 500,
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            print("Error: Unexpected response format from OpenRouter")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None


def test_llm_connectivity():
    """
    Test that the LLM API is working.
    Returns True if successful, False otherwise.
    """
    
    print("\n" + "="*60)
    print("Testing LLM Connectivity")
    print("="*60)
    
    test_prompt = "Reply with only the word: hello"
    response = call_llm(test_prompt, temperature=0)
    
    if response:
        print(f"✓ LLM is working")
        print(f"  Response: {response.strip()}")
        return True
    else:
        print("✗ LLM connection failed")
        return False


def test_guardrail():
    """
    Test that the PII guardrail is working.
    """
    
    print("\n" + "="*60)
    print("Testing PII Guardrail")
    print("="*60)
    
    # Test 1: Blocked input (email)
    blocked_input = "Email: abc@gmail.com\nAge: 40"
    if has_pii(blocked_input):
        print("✓ PII guardrail correctly blocked email")
    else:
        print("✗ PII guardrail failed to block email")
    
    # Test 2: Allowed input
    allowed_input = "Age: 40\nBMI: 31\nSmoker: Yes"
    if not has_pii(allowed_input):
        print("✓ PII guardrail correctly allowed normal input")
    else:
        print("✗ PII guardrail incorrectly blocked normal input")


def load_model():
    """
    Load the trained classification model from Part 3.
    Returns the loaded model or None if loading fails.
    """
    
    model_path = PROJECT_FOLDER / "best_model.pkl"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_test_records():
    """
    Create three handcrafted test records for prediction.
    
    Returns:
        list: List of dictionaries with feature values
    """
    
    records = [
        {
            "age": 25,
            "sex_male": 1,
            "bmi": 22.5,
            "children": 0,
            "smoker_yes": 0,
            "region_northwest": 0,
            "region_southeast": 0,
            "region_southwest": 1,
        },
        {
            "age": 45,
            "sex_male": 0,
            "bmi": 31.2,
            "children": 2,
            "smoker_yes": 1,
            "region_northwest": 0,
            "region_southeast": 1,
            "region_southwest": 0,
        },
        {
            "age": 60,
            "sex_male": 1,
            "bmi": 28.0,
            "children": 3,
            "smoker_yes": 1,
            "region_northwest": 1,
            "region_southeast": 0,
            "region_southwest": 0,
        },
    ]
    
    return records


def get_feature_names():
    """
    Return the exact feature names used by the model.
    Must match the order and names from Part 3.
    """
    
    return [
        "age",
        "sex_male",
        "bmi",
        "children",
        "smoker_yes",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]


def format_features_for_prompt(record):
    """
    Format feature values into readable text for the LLM prompt.
    """
    
    feature_text = f"""
Age: {record.get('age', 'N/A')}
Sex: {'Male' if record.get('sex_male', 0) == 1 else 'Female'}
BMI: {record.get('bmi', 'N/A')}
Children: {record.get('children', 'N/A')}
Smoker: {'Yes' if record.get('smoker_yes', 0) == 1 else 'No'}
Region: {get_region(record)}
"""
    return feature_text.strip()


def get_region(record):
    """
    Determine the region from one-hot encoded features.
    """
    
    if record.get("region_northwest", 0) == 1:
        return "Northwest"
    elif record.get("region_southeast", 0) == 1:
        return "Southeast"
    elif record.get("region_southwest", 0) == 1:
        return "Southwest"
    else:
        return "Northeast"


def make_prediction(model, record):
    """
    Use the model to make a prediction on a single record.
    
    Args:
        model: The trained classification pipeline
        record (dict): Feature values as a dictionary
    
    Returns:
        tuple: (prediction, probability)
    """
    
    feature_names = get_feature_names()
    feature_values = [[record[name] for name in feature_names]]
    
    try:
        prediction = model.predict(feature_values)[0]
        probability = model.predict_proba(feature_values)[0][1]
        return prediction, probability
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None


def explain_prediction(model, record, temperature=0):
    """
    Get an LLM explanation for a prediction.
    
    Args:
        model: The trained classification pipeline
        record (dict): Feature values
        temperature (float): LLM temperature parameter
    
    Returns:
        dict: Explanation with prediction_label, confidence_level, reasons, etc.
    """
    
    # Make the prediction
    prediction, probability = make_prediction(model, record)
    
    if prediction is None:
        return fallback()
    
    # Check for PII in the record before sending to LLM
    record_text = json.dumps(record)
    if has_pii(record_text):
        print("⚠ PII detected in record. Skipping LLM call.")
        return fallback()
    
    # Format features for the prompt
    features_text = format_features_for_prompt(record)
    
    # Determine prediction label
    prediction_label = "High Charges" if prediction == 1 else "Low Charges"
    
    # Create the user prompt
    user_prompt = USER_TEMPLATE.format(
        features=features_text,
        prediction=prediction_label,
        probability=f"{probability:.2%}"
    )
    
    # Call the LLM
    response_text = call_llm(user_prompt, temperature=temperature)
    
    if response_text is None:
        return fallback()
    
    # Validate and parse the JSON response
    result = validate_response(response_text)
    
    return result


def run_temperature_comparison(model, record):
    """
    Compare LLM responses at temperature 0 vs 0.7.
    """
    
    print("\n" + "="*60)
    print("Temperature Comparison")
    print("="*60)
    
    results = []
    
    for temp in [0, 0.7]:
        print(f"\nTemperature: {temp}")
        explanation = explain_prediction(model, record, temperature=temp)
        explanation["temperature"] = temp
        results.append(explanation)
        print(f"  Prediction Label: {explanation.get('prediction_label', 'N/A')}")
        print(f"  Confidence: {explanation.get('confidence_level', 'N/A')}")
        print(f"  Top Reason: {explanation.get('top_reason', 'N/A')}")
    
    # Save temperature comparison
    temp_df = pd.DataFrame(results)
    temp_df.to_csv(OUTPUTS_FOLDER / "temperature_comparison.csv", index=False)
    print(f"\n✓ Temperature comparison saved to temperature_comparison.csv")
    
    return results


def run_guardrail_tests():
    """
    Run comprehensive guardrail tests.
    """
    
    print("\n" + "="*60)
    print("Guardrail Test Results")
    print("="*60)
    
    results = []
    
    # Test 1: Email (should be blocked)
    test_cases = [
        {"input": "abc@gmail.com", "expected": "blocked", "name": "Email"},
        {"input": "555-123-4567", "expected": "blocked", "name": "Phone Number"},
        {"input": "Age: 40, BMI: 31", "expected": "allowed", "name": "Normal Input"},
    ]
    
    for test in test_cases:
        is_pii = has_pii(test["input"])
        expected_blocked = test["expected"] == "blocked"
        passed = is_pii == expected_blocked
        
        result_status = "PASS" if passed else "FAIL"
        action = "Blocked" if is_pii else "Allowed"
        
        print(f"{test['name']:20} | {action:10} | {result_status}")
        
        results.append({
            "test_name": test["name"],
            "input": test["input"],
            "pii_detected": is_pii,
            "expected_blocked": expected_blocked,
            "result": result_status,
        })
    
    # Save guardrail results
    guardrail_df = pd.DataFrame(results)
    guardrail_df.to_csv(OUTPUTS_FOLDER / "guardrail_results.csv", index=False)
    print(f"\n✓ Guardrail results saved to guardrail_results.csv")
    
    return results


def run_predictions(model, test_records):
    """
    Run predictions on all test records and save results.
    """
    
    print("\n" + "="*60)
    print("Running Predictions")
    print("="*60)
    
    all_results = []
    
    for idx, record in enumerate(test_records, 1):
        print(f"\nRecord {idx}:")
        
        # Make prediction
        prediction, probability = make_prediction(model, record)
        
        if prediction is None:
            print(f"  ✗ Prediction failed")
            continue
        
        prediction_label = "High Charges" if prediction == 1 else "Low Charges"
        print(f"  Prediction: {prediction_label} (p={probability:.2%})")
        
        # Get LLM explanation (temperature=0 for deterministic results)
        explanation = explain_prediction(model, record, temperature=0)
        
        # Validate the explanation
        validation_status = "PASS" if all(explanation.values()) else "FAIL"
        
        # Combine results
        result = {
            "record_id": idx,
            "prediction": prediction,
            "probability": f"{probability:.4f}",
            "prediction_label": explanation.get("prediction_label", "N/A"),
            "confidence_level": explanation.get("confidence_level", "N/A"),
            "top_reason": explanation.get("top_reason", "N/A"),
            "second_reason": explanation.get("second_reason", "N/A"),
            "next_step": explanation.get("next_step", "N/A"),
            "validation_status": validation_status,
        }
        
        all_results.append(result)
        print(f"  Validation: {validation_status}")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUTS_FOLDER / "llm_results.csv", index=False)
    print(f"\n✓ LLM results saved to llm_results.csv")
    
    # Save as JSON as well
    sample_json_path = PROJECT_FOLDER / "sample_predictions.json"
    with open(sample_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Sample predictions saved to sample_predictions.json")
    
    return all_results


def main():
    """
    Main execution function.
    """
    
    print("\n" + "="*60)
    print("PART 4: LLM-POWERED MODEL EXPLANATION PIPELINE")
    print("="*60)
    
    # Step 1: Load the model
    print("\nStep 1: Loading Model")
    print("-" * 60)
    model = load_model()
    if model is None:
        print("Fatal error: Could not load model. Exiting.")
        return
    
    # Step 2: Test LLM connectivity
    print("\nStep 2: Testing LLM Connectivity")
    print("-" * 60)
    if not test_llm_connectivity():
        print("Warning: LLM connectivity test failed. Check your API key.")
    
    # Step 3: Test guardrail
    print("\nStep 3: Testing PII Guardrail")
    print("-" * 60)
    test_guardrail()
    
    # Step 4: Run comprehensive guardrail tests
    print("\nStep 4: Guardrail Test Suite")
    print("-" * 60)
    run_guardrail_tests()
    
    # Step 5: Create test records
    print("\nStep 5: Creating Test Records")
    print("-" * 60)
    test_records = create_test_records()
    print(f"✓ Created {len(test_records)} test records")
    
    # Step 6: Run predictions
    print("\nStep 6: Running Predictions with LLM Explanations")
    print("-" * 60)
    run_predictions(model, test_records)
    
    # Step 7: Temperature comparison
    print("\nStep 7: Temperature Comparison")
    print("-" * 60)
    run_temperature_comparison(model, test_records[0])
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nOutput files saved to: {OUTPUTS_FOLDER}")
    print("  - llm_results.csv")
    print("  - temperature_comparison.csv")
    print("  - guardrail_results.csv")
    print("  - sample_predictions.json")


if __name__ == "__main__":
    main()
