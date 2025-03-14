from mainapp import embedding_model

def test_embedding_model():
    test_text = ["Hello, world!", "This is a test."]
    embeddings = embedding_model.embed_documents(test_text)
    assert len(embeddings) == len(test_text)
    assert len(embeddings[0]) > 0
