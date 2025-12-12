import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Since we are using an editable install, we can use absolute imports
from ollama_mcp.client import OllamaClient, ModelInfo

@pytest.fixture
def client():
    """Provides a fresh OllamaClient for each test."""
    return OllamaClient()

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_health_check_healthy(mock_async_client, client):
    """Test a successful health check when the Ollama server is responsive."""
    # Arrange
    mock_instance = mock_async_client.return_value
    mock_instance.ps = AsyncMock()

    # Create a mock model object that mimics the ollama library response
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {
        'model': 'test-model',
        'size': 1000,
        'digest': 'abc123',
        'modified_at': '2024-01-01T00:00:00Z'
    }

    mock_list_response = MagicMock()
    mock_list_response.models = [mock_model]
    mock_instance.list = AsyncMock(return_value=mock_list_response)

    # Act
    health = await client.health_check()

    # Assert
    assert health["healthy"] is True
    assert health["models_count"] == 1
    assert "Ollama server responsive" in health["message"]
    mock_instance.ps.assert_awaited_once()

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_health_check_unhealthy_timeout(mock_async_client, client):
    """Test an unsuccessful health check due to a timeout."""
    # Arrange
    mock_instance = mock_async_client.return_value
    mock_instance.ps.side_effect = asyncio.TimeoutError

    # Act
    health = await client.health_check()

    # Assert
    assert health["healthy"] is False
    assert "Ollama server timeout" in health["error"]

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_list_models_success(mock_async_client, client):
    """Test successfully listing models."""
    # Arrange
    mock_instance = mock_async_client.return_value

    # Create mock model objects that mimic the ollama library response
    mock_model1 = MagicMock()
    mock_model1.model_dump.return_value = {
        'model': 'llama3',
        'size': 12345,
        'digest': 'abc123',
        'modified_at': '2023-01-01T00:00:00Z'
    }

    mock_model2 = MagicMock()
    mock_model2.model_dump.return_value = {
        'model': 'qwen',
        'size': 67890,
        'digest': 'def456',
        'modified_at': '2023-01-02T00:00:00Z'
    }

    mock_list_response = MagicMock()
    mock_list_response.models = [mock_model1, mock_model2]
    mock_instance.list = AsyncMock(return_value=mock_list_response)

    # Act
    result = await client.list_models()

    # Assert
    assert result["success"] is True
    assert result["count"] == 2
    # Client now returns dicts, not ModelInfo objects
    assert isinstance(result["models"][0], dict)
    assert result["models"][0]["name"] == 'llama3'
    assert result["models"][0]["size_human"] == '12.1 KB'

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_list_models_api_error(mock_async_client, client):
    """Test handling of an API error when listing models."""
    # Arrange
    mock_instance = mock_async_client.return_value
    mock_instance.list.side_effect = Exception("API connection failed")

    # Act
    result = await client.list_models()

    # Assert
    assert result["success"] is False
    assert "API connection failed" in result["error"]
    assert result["models"] == []

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_chat_success(mock_async_client, client):
    """Test a successful chat session."""
    # Arrange
    mock_instance = mock_async_client.return_value
    mock_chat_response = {
        'message': {'content': 'This is a test response.'},
        'eval_count': 10,
        'total_duration': 12345
    }
    mock_instance.chat = AsyncMock(return_value=mock_chat_response)

    # Act
    result = await client.chat("test-model", "Hello")

    # Assert
    assert result["success"] is True
    assert result["response"] == 'This is a test response.'
    assert result["model"] == "test-model"
    mock_instance.chat.assert_awaited_once()

@pytest.mark.asyncio
@patch('ollama.AsyncClient')
async def test_remove_model_not_found(mock_async_client, client):
    """Test attempting to remove a model that does not exist."""
    # Arrange
    import ollama
    mock_instance = mock_async_client.return_value
    mock_instance.delete.side_effect = ollama.ResponseError("Model not found", status_code=404)

    # Act
    result = await client.remove_model("nonexistent-model")

    # Assert
    assert result["success"] is False
    assert "not found" in result["error"]

def test_model_info_size_human_readable():
    """Test the human-readable size conversion in ModelInfo."""
    # Test cases for various sizes - ModelInfo requires 4 args: name, size, digest, modified_at
    assert ModelInfo("test", 100, "", "2024-01-01").size_human == "100.0 B"
    assert ModelInfo("test", 2 * 1024, "", "2024-01-01").size_human == "2.0 KB"
    assert ModelInfo("test", 3 * 1024**2, "", "2024-01-01").size_human == "3.0 MB"
    assert ModelInfo("test", 4 * 1024**3, "", "2024-01-01").size_human == "4.0 GB"
    assert ModelInfo("test", 5 * 1024**4, "", "2024-01-01").size_human == "5.0 TB"
