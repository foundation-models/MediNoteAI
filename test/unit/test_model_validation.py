import pytest
import json
from unittest.mock import Mock, patch
from pydantic import BaseModel
from typing import Optional

# Import the functions we want to test
from MediNoteAI.test.integration.validation_via_model import (
    extract_response_content,
    call_extraction_model_all,
    extract_params,
)

# Test model for validation
class TestModel(BaseModel):
    name: str
    location: str
    industry: Optional[str] = None

@pytest.fixture
def mock_response():
    """Fixture for mocking different types of response objects"""
    class MockResponse:
        def __init__(self, content_type="direct"):
            if content_type == "direct":
                self.content = "test content"
            elif content_type == "message":
                self.message = Mock(content="test message content")
            elif content_type == "text":
                self.text = "test text"
            elif content_type == "choices":
                self.choices = [Mock(message=Mock(content="test choice content"))]

    return MockResponse

@pytest.mark.asyncio
async def test_extract_response_content_direct():
    """Test extracting content from response with direct content attribute"""
    mock_resp = Mock(content="test content")
    result = extract_response_content(mock_resp)
    assert result == "test content"

@pytest.mark.asyncio
async def test_extract_response_content_message():
    """Test extracting content from response with message.content structure"""
    mock_resp = Mock()
    mock_resp.message = Mock(content="test message content")
    result = extract_response_content(mock_resp)
    assert result == "test message content"

@pytest.mark.asyncio
async def test_call_extraction_model_all():
    """Test the extraction of fields from user input"""
    user_input = "Company XYZ in New York works in technology"
    fields = ["name", "location", "industry"]
    
    # Mock the model client response
    mock_response = {
        "name": "Company XYZ",
        "location": "New York",
        "industry": "technology"
    }
    
    with patch('autogen_ext.models.openai.AzureOpenAIChatCompletionClient.create') as mock_create:
        # Setup the mock to return a response object with the expected content
        mock_create.return_value = Mock(content=json.dumps(mock_response))
        
        result = await call_extraction_model_all(user_input, fields)
        
        assert isinstance(result, dict)
        assert result["name"] == "Company XYZ"
        assert result["location"] == "New York"
        assert result["industry"] == "technology"

@pytest.mark.asyncio
async def test_extract_params_success():
    """Test successful parameter extraction and model creation"""
    user_input = "Company XYZ in New York works in technology"
    
    # Mock the extraction response
    mock_response = {
        "name": "Company XYZ",
        "location": "New York",
        "industry": "technology"
    }
    
    with patch('MediNoteAI.test.integration.validation_via_model.call_extraction_model_all') as mock_extract:
        mock_extract.return_value = mock_response
        
        model_instance, messages = await extract_params(user_input, TestModel)
        
        assert model_instance is not None
        assert isinstance(model_instance, TestModel)
        assert model_instance.name == "Company XYZ"
        assert model_instance.location == "New York"
        assert model_instance.industry == "technology"
        assert all(isinstance(msg, str) for msg in messages.values())

@pytest.mark.asyncio
async def test_extract_params_missing_required():
    """Test parameter extraction with missing required fields"""
    user_input = "Some company works in technology"
    
    # Mock response with missing required field
    mock_response = {
        "name": "",  # Missing required field
        "location": "Unknown",
        "industry": "technology"
    }
    
    with patch('MediNoteAI.test.integration.validation_via_model.call_extraction_model_all') as mock_extract:
        mock_extract.return_value = mock_response
        
        model_instance, messages = await extract_params(user_input, TestModel)
        
        assert model_instance is None  # Should be None due to validation error
        assert isinstance(messages, dict)
        assert any("name" in msg.lower() for msg in messages.values())  # Should contain error about name

@pytest.mark.asyncio
async def test_extract_params_empty_input():
    """Test parameter extraction with empty input"""
    user_input = ""
    
    model_instance, messages = await extract_params(user_input, TestModel)
    
    assert model_instance is None
    assert isinstance(messages, dict)
    assert len(messages) > 0  # Should contain error messages

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 