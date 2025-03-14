from unittest.mock import MagicMock
from mainapp import llm

def test_chatbot_response():
    llm.generate_content = MagicMock(return_value=MagicMock(text="Mocked response"))

    response = llm.generate_content("What is Encoder stack?")
    
    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.text == "Mocked response"