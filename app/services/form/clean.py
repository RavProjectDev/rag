#!/usr/bin/env python3
"""
Script to convert JSON file from list format to dictionary format
where each key is the user question.
"""

import json
import sys
from pathlib import Path


def convert_json_list_to_dict(input_file, output_file=None):
    """
    Convert JSON file from list format to dictionary format.

    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Path to output JSON file. If None,
                                   creates output file with '_dict' suffix
    """

    # Read the input JSON file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        return False

    # Validate that data is a list
    if not isinstance(data, list):
        print("Error: JSON file should contain a list of objects.")
        return False

    # Convert list to dictionary
    result_dict = {}

    for i, item in enumerate(data):
        # Validate each item has the expected structure
        if not isinstance(item, dict):
            print(f"Warning: Item {i} is not a dictionary, skipping...")
            continue

        if "user_question" not in item:
            print(f"Warning: Item {i} missing 'user_question' field, skipping...")
            continue

        user_question = item["user_question"]

        # Handle duplicate questions by adding a suffix
        original_question = user_question
        counter = 1
        while user_question in result_dict:
            user_question = f"{original_question} ({counter})"
            counter += 1

        # Store the item with user_question as key
        # Remove user_question from the stored data since it's now the key
        item_copy = item.copy()
        del item_copy["user_question"]
        result_dict[user_question] = item_copy

    # Determine output file name
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_dict{input_path.suffix}"

    # Write the converted data to output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"Successfully converted {len(result_dict)} items.")
        print(f"Output saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error writing to '{output_file}': {e}")
        return False


def main():

    input_file = "/Users/dothanbardichev/Desktop/RAV/RAG/rag/app/form_data/questions_answers_cleaned.json"  # Replace with your JSON file path
    json_output_file = "/Users/dothanbardichev/Desktop/RAV/RAG/rag/app/form_data/questions_answers_cleaned.json"  # JSON o

    success = convert_json_list_to_dict(input_file, json_output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
