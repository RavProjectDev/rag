import json


def verify(event) -> tuple[bool, str]:
    try:
        question = event.get("question")
        if not question:
            return False, "body needs to include question"

        return True, question

    except json.JSONDecodeError:
        return False, "body needs to be in json format"
    except ValueError as ve:
        return False, str(ve)
    except Exception as e:
        return False, str(e)
