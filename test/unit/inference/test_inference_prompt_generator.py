import os
import requests
import json
import pytest
from unittest.mock import patch
from fuzzywuzzy import fuzz

import yaml
from medinote.cached import read_dataframe
from medinote.inference.inference_prompt_generator import parallel_row_infer, row_infer
from medinote import initialize

logger, _ = initialize()

def test_row_infer():
    # Mocking the requests.post method
    with patch.object(requests, 'post') as mock_post:
        # Mocking the response from the inference API
        mock_response = {
            "text": "Inference result"
        }
        mock_post.return_value.json.return_value = mock_response

        # Test data
        row = {
            "column1": "value1",
            "column2": "value2"
        }
        conf = {
            "prompt_template": "This is a prompt: {column1}, {column2}",
            "prompt_column": "prompt",
            "payload_template": "{{\"prompt\": \"{prompt}\"}}",
            "inference_url": "http://phi-generative-ai:8888/worker_generate"
        }

        # Expected result
        expected_result = {
            "column1": "value1",
            "column2": "value2",
            "prompt": "This is a prompt: value1, value2",
            "response": "Inference result"
        }

        # Call the function under test
        result = row_infer(row, conf)

        # Assertions
        assert result == expected_result
        mock_post.assert_called_once_with(url="http://phi-generative-ai:8888/worker_generate", headers={"Content-Type": "application/json"}, json={"prompt": "This is a prompt: value1, value2"})
        assert result["response"] == "Inference result"
        
with open(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../../config/config.yaml", "r"
) as file:
    conf = yaml.safe_load(file).get("test")        

def test_integration_row_infer():
    row = {
        "Narrative": "Attended a meeting with Matt Klein to discuss the acquisition target. Drafted an LOI document. Conducted due diligence research on the target company."
    }

    # Call the function under test
    row = row_infer(row, conf)
    response = row.get("inference_response")

    # Assertions
    # expected_response = "Corporation engaged Law Firm's services in the amount of X hrs. during the week of [date1] for attending a meeting with Matt Klein to discuss the acquisition target and drafting of the Letter of Intent (LOI) document. During the week of [date2], the Corporation was additionally engaged Law Firm's services for conducting due diligence research on the target company."
    expected_response = 'Assistant has attended meeting with Matt Klein to discuss acquisition target. Drafted Letter of Intent document for acquisition. '
    similarity_threshold = 50  # percentage
    similarity = fuzz.partial_ratio(expected_response, response)
    logger.info(f"Response:\n{response}\n\nExpected Result:\n{expected_response} \n\nSimilarity:\n{similarity}")
    assert similarity >= similarity_threshold
    
def test_integration_parallel_row_infer():
    test_dataset_path = conf.get('test_dataset_path')
    
    if test_dataset_path is None:
        raise ValueError("Test dataset path not found in config file")
    
    test_df = read_dataframe(test_dataset_path)
    test_df = test_df.head(10)
    df = parallel_row_infer(df=test_df, persist=False, config=conf)
    assert df is not None
    assert len(df) == 10
        
if __name__ == "__main__":
    test_integration_parallel_row_infer()       
                
        
       