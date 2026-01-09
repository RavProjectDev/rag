"""
Unit tests for spaCy-based semantic chunking.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestSpacySentenceSplitting:
    """Test spaCy sentence splitting functionality."""
    
    def test_split_into_sentences_basic(self):
        """Test basic sentence splitting with spaCy."""
        from rag.app.services.preprocess.chunking_strategies.semantic import split_into_sentences
        
        # Mock spaCy to avoid downloading model during tests
        with patch('rag.app.services.preprocess.chunking_strategies.semantic._get_spacy_model') as mock_nlp_getter:
            # Create mock spaCy doc and sentences
            mock_sent1 = MagicMock()
            mock_sent1.text = "This is the first sentence."
            
            mock_sent2 = MagicMock()
            mock_sent2.text = "This is the second sentence."
            
            mock_sent3 = MagicMock()
            mock_sent3.text = "This is the third sentence."
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3]
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_nlp_getter.return_value = mock_nlp
            
            text = "This is the first sentence. This is the second sentence. This is the third sentence."
            sentences = split_into_sentences(text)
            
            assert len(sentences) == 3
            assert sentences[0] == "This is the first sentence."
            assert sentences[1] == "This is the second sentence."
            assert sentences[2] == "This is the third sentence."
    
    def test_split_into_sentences_with_abbreviations(self):
        """Test sentence splitting handles abbreviations correctly."""
        from rag.app.services.preprocess.chunking_strategies.semantic import split_into_sentences
        
        with patch('rag.app.services.preprocess.chunking_strategies.semantic._get_spacy_model') as mock_nlp_getter:
            # Mock sentences that would be harder for regex
            mock_sent1 = MagicMock()
            mock_sent1.text = "Dr. Smith works at U.S.A."
            
            mock_sent2 = MagicMock()
            mock_sent2.text = "The temperature is 98.6°F."
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent1, mock_sent2]
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_nlp_getter.return_value = mock_nlp
            
            text = "Dr. Smith works at U.S.A. The temperature is 98.6°F."
            sentences = split_into_sentences(text)
            
            assert len(sentences) == 2
            assert "Dr. Smith" in sentences[0]
            assert "98.6" in sentences[1]
    
    def test_split_into_sentences_empty_text(self):
        """Test sentence splitting with empty text."""
        from rag.app.services.preprocess.chunking_strategies.semantic import split_into_sentences
        
        sentences = split_into_sentences("")
        assert sentences == []
        
        sentences = split_into_sentences("   ")
        assert sentences == []
    
    def test_split_into_sentences_single_sentence(self):
        """Test sentence splitting with single sentence."""
        from rag.app.services.preprocess.chunking_strategies.semantic import split_into_sentences
        
        with patch('rag.app.services.preprocess.chunking_strategies.semantic._get_spacy_model') as mock_nlp_getter:
            mock_sent = MagicMock()
            mock_sent.text = "This is a single sentence"
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent]
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_nlp_getter.return_value = mock_nlp
            
            text = "This is a single sentence"
            sentences = split_into_sentences(text)
            
            assert len(sentences) == 1
            assert sentences[0] == "This is a single sentence"


class TestSpacyModelLoader:
    """Test spaCy model loading functionality."""
    
    def test_get_spacy_model_cached(self):
        """Test that spaCy model is cached properly."""
        from rag.app.services.preprocess.chunking_strategies.semantic import _get_spacy_model
        
        # Clear cache before test
        _get_spacy_model.cache_clear()
        
        with patch('spacy.load') as mock_load:
            mock_nlp = MagicMock()
            mock_nlp.select_pipes = MagicMock()
            mock_load.return_value = mock_nlp
            
            # First call
            nlp1 = _get_spacy_model()
            # Second call should use cache
            nlp2 = _get_spacy_model()
            
            # Should only load once due to caching
            assert mock_load.call_count == 1
            assert nlp1 is nlp2
        
        # Clear cache after test
        _get_spacy_model.cache_clear()


@pytest.mark.asyncio
class TestSemanticChunking:
    """Test semantic chunking with spaCy integration."""
    
    async def test_build_semantic_chunks_integration(self):
        """Test semantic chunking creates appropriate chunks with token counting."""
        from rag.app.services.preprocess.chunking_strategies.semantic import build_semantic_chunks
        from rag.app.schemas.data import EmbeddingConfiguration
        
        # Mock spaCy, embedding generation, and token counting
        with patch('rag.app.services.preprocess.chunking_strategies.semantic._get_spacy_model') as mock_nlp_getter, \
             patch('rag.app.services.preprocess.chunking_strategies.semantic._generate_sentence_embedding_for_chunking') as mock_embed, \
             patch('rag.app.services.preprocess.chunking_strategies.semantic._count_tokens') as mock_count_tokens:
            
            # Mock spaCy sentences
            mock_sent1 = MagicMock()
            mock_sent1.text = "First sentence about topic A."
            
            mock_sent2 = MagicMock()
            mock_sent2.text = "Second sentence about topic A."
            
            mock_sent3 = MagicMock()
            mock_sent3.text = "Third sentence about topic B."
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3]
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_nlp_getter.return_value = mock_nlp
            
            # Mock embeddings - similar for first two, different for third
            import numpy as np
            mock_embed.side_effect = [
                np.random.rand(384).tolist(),  # Sentence 1
                np.random.rand(384).tolist(),  # Sentence 2
                np.random.rand(384).tolist(),  # Sentence 3
            ]
            
            # Mock token counting - return reasonable token counts
            mock_count_tokens.side_effect = lambda text: max(50, len(text.split()) * 2)
            
            embedding_config = EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE
            
            text = "First sentence about topic A. Second sentence about topic A. Third sentence about topic B."
            chunks = await build_semantic_chunks(
                text=text,
                name_space="test_doc",
                embedding_configuration=embedding_config,
                similarity_threshold=0.7
            )
            
            # Should create at least one chunk
            assert len(chunks) > 0
            
            # All chunks should have the namespace and token-based sizes
            for chunk in chunks:
                assert chunk.name_space == "test_doc"
                assert chunk.chunk_size > 0  # Token count, not word count
                assert len(chunk.text_to_embed) > 0


class TestTokenCounting:
    """Test token counting functionality."""
    
    def test_count_tokens(self):
        """Test that token counting works correctly."""
        from rag.app.services.preprocess.chunking_strategies.semantic import _count_tokens
        
        # Mock the tokenizer
        with patch('rag.app.services.preprocess.chunking_strategies.semantic._get_tokenizer') as mock_get_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_tokenizer.return_value = mock_tokenizer
            
            text = "This is a test sentence."
            token_count = _count_tokens(text)
            
            assert token_count == 5
            mock_tokenizer.encode.assert_called_once_with(text)
    
    def test_count_tokens_empty_text(self):
        """Test token counting with empty text."""
        from rag.app.services.preprocess.chunking_strategies.semantic import _count_tokens
        
        assert _count_tokens("") == 0
        assert _count_tokens(None) == 0

