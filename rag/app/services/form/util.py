def load_and_convert_from_json(json_file_path, output_file_path=None):
    """
    Loads JSON data from file and converts dictionary from format:
    {
        "some_question": ["response1", "response2", "response3"],
        ...
    }

    To format:
    {
        "some_question": {
            "responses": [
                {"prompt_id": "1", "llm_response": "response1"},
                {"prompt_id": "2", "llm_response": "response2"},
                {"prompt_id": "3", "llm_response": "response3"}
            ]
        },
        ...
    }

    If output_file_path is provided, writes the converted data to that file.
    """
    import json

    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            input_dict = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Error reading file '{json_file_path}': {e}")
        return {}

    converted_dict = {}

    for question, responses in input_dict.items():
        converted_dict[question] = {
            "responses": [
                {"prompt_id": str(i + 1), "llm_response": response}
                for i, response in enumerate(responses)
            ]
        }

    # Write to output file if specified
    if output_file_path:
        try:
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(converted_dict, output_file, indent=2, ensure_ascii=False)
            print(f"Converted data successfully written to: {output_file_path}")
        except Exception as e:
            print(f"Error writing to file '{output_file_path}': {e}")

    return converted_dict


def save_converted_data(converted_dict, output_file_path):
    """
    Saves the converted dictionary to a JSON file.
    """
    import json

    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(converted_dict, output_file, indent=2, ensure_ascii=False)
        print(f"Converted data successfully written to: {output_file_path}")
        return True
    except Exception as e:
        print(f"Error writing to file '{output_file_path}': {e}")
        return False


def convert_dictionary_format(input_dict):
    """
    Converts dictionary from format (for when you already have the dict loaded):
    {
        "some_question": ["response1", "response2", "response3"],
        ...
    }

    To format:
    {
        "some_question": {
            "responses": [
                {"prompt_id": "1", "llm_response": "response1"},
                {"prompt_id": "2", "llm_response": "response2"},
                {"prompt_id": "3", "llm_response": "response3"}
            ]
        },
        ...
    }
    """
    converted_dict = {}

    for question, responses in input_dict.items():
        converted_dict[question] = {
            "responses": [
                {"prompt_id": str(i + 1), "llm_response": response}
                for i, response in enumerate(responses)
            ]
        }

    return converted_dict


def get_question_data(form_data_dict, question):
    """
    Helper function to get question data in the format you specified:
    return form_data.data.data.get(question, "")

    This assumes form_data_dict is the converted dictionary.
    """
    return form_data_dict.get(question, "")


# Example usage
if __name__ == "__main__":
    import json

    # INSERT YOUR JSON FILE PATHS HERE
    input_json_file_path = "/Users/dothanbardichev/Desktop/RAV/RAG/rag/app/form_data/questions_llm_response_map.json"  # <-- Change this to your input JSON file path
    output_json_file_path = "/Users/dothanbardichev/Desktop/RAV/RAG/rag/app/form_data/questions_answers_cleaned.json"  # <-- Change this to your output JSON file path

    # Option 1: Load, convert, and save in one step
    converted_data = load_and_convert_from_json(
        input_json_file_path, output_json_file_path
    )

    # Option 2: Load and convert first, then save separately (alternative approach)
    # converted_data = load_and_convert_from_json(input_json_file_path)
    # if converted_data:
    #     save_converted_data(converted_data, output_json_file_path)

    if converted_data:  # Only proceed if loading was successful
        # Print the result to console as well
        print("\nConverted dictionary preview:")
        print(json.dumps(converted_data, indent=2))

        # Example of accessing data
        print("\n" + "=" * 50)
        print("Example access:")

        # Get first question from the converted data (if any exist)
        if converted_data:
            first_question = list(converted_data.keys())[0]
            result = get_question_data(converted_data, first_question)
            print(f"Question: {first_question}")
            print(f"Data: {result}")

            # If you want to access individual responses:
            if result:
                print("\nIndividual responses:")
                for response_obj in result["responses"]:
                    print(
                        f"Response {response_obj['prompt_id']}: {response_obj['llm_response']}"
                    )
    else:
        print("Failed to load or convert data from JSON file.")
