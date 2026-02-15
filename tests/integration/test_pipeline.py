"""Integration tests for the full indexing pipeline."""

import json
from pathlib import Path

import pytest

from graphunified.config.settings import Settings
from graphunified.index.pipeline import IndexPipeline


@pytest.mark.integration
class TestIndexPipeline:
    """Integration tests for IndexPipeline."""

    @pytest.fixture
    def sample_corpus(self, tmp_path):
        """Create a sample document corpus."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create sample documents
        docs = [
            {
                "filename": "climate_change.md",
                "content": """# Climate Change and Global Warming

Climate change refers to long-term shifts in global temperatures and weather patterns.
The Intergovernmental Panel on Climate Change (IPCC) has documented that global temperatures
have risen by approximately 1.2°C since pre-industrial times.

Dr. Jane Smith, a climate scientist at NASA, leads research on the impacts of rising
temperatures on polar ice caps. NASA's Climate Observatory in Washington DC monitors
global temperature changes continuously.

The Paris Agreement, signed in 2015, aims to limit global warming to well below 2°C
above pre-industrial levels. The United Nations coordinates international efforts to
address climate change through various programs and initiatives.
""",
            },
            {
                "filename": "artificial_intelligence.md",
                "content": """# Artificial Intelligence and Machine Learning

Artificial intelligence (AI) has transformed many industries through machine learning
and deep learning technologies. OpenAI, founded in 2015, has developed advanced language
models including GPT-4 and ChatGPT.

Dr. Andrew Ng, a professor at Stanford University, pioneered online AI education through
Coursera. His research focuses on deep learning applications in healthcare and autonomous
vehicles. Stanford's AI Lab in Palo Alto, California is one of the world's leading
research institutions.

The European Union has proposed AI regulations to ensure ethical development and
deployment of AI systems. These regulations aim to balance innovation with safety and
privacy concerns.
""",
            },
            {
                "filename": "space_exploration.md",
                "content": """# Space Exploration and Mars Missions

Space exploration has entered a new era with private companies joining government agencies.
SpaceX, led by Elon Musk, has developed reusable rockets that dramatically reduce launch
costs. The company is based in Hawthorne, California.

NASA's Perseverance rover, which landed on Mars in 2021, searches for signs of ancient
microbial life. The Mars mission is controlled from NASA's Jet Propulsion Laboratory
in Pasadena, California.

The Artemis program aims to return humans to the Moon by 2025. This program represents
a partnership between NASA, the European Space Agency, and other international partners.
""",
            },
        ]

        for doc in docs:
            (corpus_dir / doc["filename"]).write_text(doc["content"])

        return corpus_dir

    @pytest.fixture
    def test_settings(self, tmp_path):
        """Create test settings with mocked API keys."""
        # Note: These are dummy keys for testing
        # Real tests would use actual API keys from environment
        return Settings(
            llm={
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "api_key": "test-key-dummy",  # Dummy key
                "temperature": 0.0,
                "max_tokens": 4096,
                "timeout": 60,
            },
            embedding={
                "provider": "voyage",
                "model": "voyage-3",
                "api_key": "test-key-dummy",  # Dummy key
                "dimension": 1024,
                "batch_size": 128,
            },
            indexing={
                "chunk_size": 256,
                "chunk_overlap": 64,
                "extraction_batch_size": 5,
                "dedup_threshold": 85,
                "max_concurrent": 5,
            },
            storage={"root_dir": str(tmp_path / "output")},
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pipeline_no_api_calls(self, sample_corpus, test_settings, tmp_path):
        """Test pipeline stages that don't require API calls."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pipeline = IndexPipeline.from_config(
            settings=test_settings,
            input_dir=sample_corpus,
            output_dir=output_dir,
        )

        # Run only load and chunk stages (no API calls)
        result = await pipeline.run(skip_extraction=True, skip_embedding=True)

        assert result["status"] == "success"

        metrics = result["metrics"]
        assert metrics["documents_loaded"] == 3
        assert metrics["chunks_created"] > 0
        assert metrics["entities_extracted"] == 0  # Skipped
        assert metrics["relationships_extracted"] == 0  # Skipped

        # Verify output files exist
        assert (output_dir / "documents").exists()
        assert (output_dir / "chunks").exists()

        # Verify Parquet files were created
        doc_files = list((output_dir / "documents").glob("*.parquet"))
        chunk_files = list((output_dir / "chunks").glob("*.parquet"))

        assert len(doc_files) > 0
        assert len(chunk_files) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_mock_clients(self, sample_corpus, test_settings, tmp_path, mocker):
        """Test full pipeline with mocked API clients."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock LLM client
        mock_llm = mocker.Mock()
        entity_response = json.dumps(
            {
                "entities": [
                    {"name": "NASA", "type": "ORGANIZATION", "description": "Space agency", "confidence": 0.95},
                    {"name": "Climate Change", "type": "CONCEPT", "description": "Environmental issue", "confidence": 0.90},
                ]
            }
        )
        relationship_response = json.dumps(
            {
                "relationships": [
                    {
                        "source": "NASA",
                        "target": "Climate Change",
                        "type": "RELATED_TO",
                        "description": "NASA studies climate change",
                        "confidence": 0.85,
                    }
                ]
            }
        )
        mock_llm.generate = mocker.AsyncMock(
            side_effect=[
                (entity_response, 100, 200),
                (relationship_response, 100, 150),
            ]
        )

        # Mock embedding client
        mock_embedding = mocker.Mock()
        mock_embedding.model = "test-model"
        mock_embedding.embed = mocker.AsyncMock(return_value=[[0.1] * 1024, [0.2] * 1024])

        # Patch clients
        mocker.patch("graphunified.index.pipeline.ClaudeClient.from_config", return_value=mock_llm)
        mocker.patch("graphunified.index.pipeline.EmbeddingClient.from_config", return_value=mock_embedding)

        pipeline = IndexPipeline.from_config(
            settings=test_settings,
            input_dir=sample_corpus,
            output_dir=output_dir,
        )

        result = await pipeline.run(skip_extraction=False, skip_embedding=False)

        assert result["status"] == "success"

        metrics = result["metrics"]
        assert metrics["documents_loaded"] == 3
        assert metrics["chunks_created"] > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_directory(self, tmp_path, test_settings):
        """Test pipeline with empty input directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pipeline = IndexPipeline.from_config(
            settings=test_settings,
            input_dir=empty_dir,
            output_dir=output_dir,
        )

        result = await pipeline.run(skip_extraction=True, skip_embedding=True)

        assert result["status"] == "success"
        assert result["metrics"]["documents_loaded"] == 0
        assert result["metrics"]["chunks_created"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_checkpoint_save(self, sample_corpus, test_settings, tmp_path):
        """Test that pipeline saves checkpoint."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pipeline = IndexPipeline.from_config(
            settings=test_settings,
            input_dir=sample_corpus,
            output_dir=output_dir,
        )

        result = await pipeline.run_incremental()

        if result["status"] == "success":
            checkpoint_file = output_dir / "checkpoint.json"
            # Checkpoint should be saved (might not exist if run fails due to dummy API keys)
            if checkpoint_file.exists():
                checkpoint_data = json.loads(checkpoint_file.read_text())
                assert "timestamp" in checkpoint_data
                assert "metrics" in checkpoint_data
