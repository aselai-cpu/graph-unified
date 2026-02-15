"""Command-line interface for graph-unified."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from graphunified.config.settings import Settings
from graphunified.exceptions import ConfigurationError
from graphunified.index.pipeline import IndexPipeline
from graphunified.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="graph-unified")
def cli():
    """Graph-Unified: Multi-Strategy RAG System

    A unified RAG system supporting Naive, Hybrid, GraphRAG, LightRAG, and HippoRAG strategies.
    """
    pass


@cli.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Input directory containing documents (.txt, .md)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for indexed data (Parquet files)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="settings.yaml",
    help="Configuration file path (YAML)",
)
@click.option(
    "--skip-extraction",
    is_flag=True,
    help="Skip entity/relationship extraction (for testing)",
)
@click.option(
    "--skip-embedding",
    is_flag=True,
    help="Skip embedding generation (for testing)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level)",
)
def index(
    input_dir: Path,
    output_dir: Path,
    config: Path,
    skip_extraction: bool,
    skip_embedding: bool,
    verbose: bool,
):
    """Index documents and extract knowledge graph.

    This command processes documents through the full indexing pipeline:

    \b
    1. Load documents from INPUT_DIR
    2. Chunk documents into overlapping windows
    3. Extract entities and relationships using Claude
    4. Generate embeddings for chunks and entities using Voyage AI
    5. Save results to OUTPUT_DIR as Parquet files

    Example:

        graph-unified index -i ./corpus -o ./output -c settings.yaml
    """
    # Setup logging
    from graphunified.config.settings import LoggingConfig
    log_level = "DEBUG" if verbose else "INFO"
    log_config = LoggingConfig(level=log_level, format="text", output="stdout")
    setup_logging(log_config)

    logger.info("=" * 80)
    logger.info("Graph-Unified Indexing Pipeline")
    logger.info("=" * 80)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {config}")
        settings = Settings.load(config)
        settings.validate_completeness()

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Create progress bar
        progress_bar = None
        if not verbose:  # Only show progress bar in non-verbose mode
            progress_bar = tqdm(total=100, desc="Indexing", unit="%", ncols=80)

        def progress_callback(stage: str, progress: float):
            """Update progress bar"""
            if progress_bar:
                progress_bar.set_description(f"{stage}")
                progress_bar.n = int(progress * 100)
                progress_bar.refresh()

        # Create and run pipeline
        pipeline = IndexPipeline.from_config(
            settings=settings,
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback if progress_bar else None,
        )

        # Run async pipeline
        result = asyncio.run(
            pipeline.run(skip_extraction=skip_extraction, skip_embedding=skip_embedding)
        )

        if progress_bar:
            progress_bar.n = 100
            progress_bar.close()

        # Display results
        if result["status"] == "success":
            metrics = result["metrics"]
            logger.info("")
            logger.info("=" * 80)
            logger.info("Indexing Complete!")
            logger.info("=" * 80)
            logger.info(f"Documents processed: {metrics['documents_loaded']}")
            logger.info(f"Chunks created: {metrics['chunks_created']}")
            logger.info(f"Entities extracted: {metrics['entities_extracted']}")
            logger.info(f"Relationships extracted: {metrics['relationships_extracted']}")
            logger.info(f"Chunks with embeddings: {metrics['chunks_with_embeddings']}")
            logger.info(f"Entities with embeddings: {metrics['entities_with_embeddings']}")
            logger.info(f"Duration: {metrics['duration_seconds']:.2f}s")
            logger.info(f"Output: {metrics['output_dir']}")
            logger.info("=" * 80)
            sys.exit(0)
        else:
            logger.error(f"Indexing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--query",
    "-q",
    type=str,
    required=True,
    help="Query text",
)
@click.option(
    "--index-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Index directory (output from index command)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="settings.yaml",
    help="Configuration file path (YAML)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(
        ["naive", "hybrid", "graphrag_local", "graphrag_global", "lightrag", "hipporag", "auto"],
        case_sensitive=False,
    ),
    default="auto",
    help="Retrieval strategy to use",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=10,
    help="Number of results to retrieve",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def query(
    query: str,
    index_dir: Path,
    config: Path,
    strategy: str,
    top_k: int,
    verbose: bool,
):
    """Query the indexed knowledge graph.

    This command retrieves relevant information using the specified strategy.

    Example:

        graph-unified query -q "What is climate change?" -i ./output -s hybrid
    """
    # Setup logging
    from graphunified.config.settings import LoggingConfig
    log_level = "DEBUG" if verbose else "INFO"
    log_config = LoggingConfig(level=log_level, format="text", output="stdout")
    setup_logging(log_config)

    logger.info("Query functionality will be implemented in Phase 3")
    logger.info(f"Query: {query}")
    logger.info(f"Index: {index_dir}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Top-K: {top_k}")

    click.echo("Query command is not yet implemented. Coming in Phase 3!")
    sys.exit(0)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
