from medinote.curation.rest_clients import call_openai

def test_call_openai():
    prompt = "Hello, AI!"
    instruction = "Please generate a response."
    expected_result = {
        "response": "This is the generated response."
    }

    result = call_openai(prompt, instruction)
    assert result == expected_result

if __name__ == "__main__":
    test_call_openai()
    print("All tests passed!")