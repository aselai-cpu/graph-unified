"""Verification script for GraphRAG Global enhancements.

This script checks that all three enhancements are properly implemented:
1. Embedding-based community search
2. Map-reduce answer synthesis
3. Leiden algorithm
"""

import ast
import inspect
from pathlib import Path


def check_imports():
    """Verify required imports are present."""
    print("=" * 80)
    print("1. Checking Imports")
    print("=" * 80)

    graphrag_global_path = Path("graphunified/strategies/graphrag_global.py")
    content = graphrag_global_path.read_text()

    required_imports = [
        "from graphunified.storage.vector_store import VectorStore",
        "from typing import Any, Dict, List, Set, Tuple",
        "from uuid import UUID, uuid4",
    ]

    for imp in required_imports:
        if imp in content:
            print(f"✅ Found: {imp}")
        else:
            print(f"❌ Missing: {imp}")
    print()


def check_methods():
    """Verify new methods exist."""
    print("=" * 80)
    print("2. Checking Methods")
    print("=" * 80)

    graphrag_global_path = Path("graphunified/strategies/graphrag_global.py")
    content = graphrag_global_path.read_text()

    required_methods = [
        "_rank_communities_semantic",
        "_rank_communities_keyword",
        "_synthesize_answer",
    ]

    for method in required_methods:
        if f"async def {method}" in content or f"def {method}" in content:
            print(f"✅ Method exists: {method}")
        else:
            print(f"❌ Method missing: {method}")
    print()


def check_leiden_algorithm():
    """Verify Leiden algorithm is used."""
    print("=" * 80)
    print("3. Checking Leiden Algorithm")
    print("=" * 80)

    graphrag_global_path = Path("graphunified/strategies/graphrag_global.py")
    content = graphrag_global_path.read_text()

    checks = [
        ("detect_communities_leiden", "✅ Leiden method called"),
        ("falling back to Louvain", "✅ Fallback logic present"),
        ("Using Leiden algorithm", "✅ Leiden logging present"),
    ]

    for pattern, message in checks:
        if pattern in content:
            print(message)
        else:
            print(f"❌ Missing: {pattern}")
    print()


def check_vector_store_integration():
    """Verify VectorStore integration."""
    print("=" * 80)
    print("4. Checking VectorStore Integration")
    print("=" * 80)

    graphrag_global_path = Path("graphunified/strategies/graphrag_global.py")
    content = graphrag_global_path.read_text()

    checks = [
        ("vector_store: VectorStore", "✅ VectorStore parameter in __init__"),
        ("self.vector_store = vector_store", "✅ VectorStore stored as instance variable"),
        ("await self.vector_store.index_communities", "✅ Community indexing called"),
        ("await self.vector_store.search_communities", "✅ Community search called"),
    ]

    for pattern, message in checks:
        if pattern in content:
            print(message)
        else:
            print(f"❌ Missing: {pattern}")
    print()


def check_synthesis_in_retrieve():
    """Verify synthesis is integrated into retrieve method."""
    print("=" * 80)
    print("5. Checking Synthesis Integration")
    print("=" * 80)

    graphrag_global_path = Path("graphunified/strategies/graphrag_global.py")
    content = graphrag_global_path.read_text()

    checks = [
        ("await self._rank_communities_semantic", "✅ Semantic ranking called"),
        ("await self._synthesize_answer", "✅ Synthesis called"),
        ("is_synthesized", "✅ Synthesis metadata present"),
        ("source_communities", "✅ Source tracking metadata"),
    ]

    for pattern, message in checks:
        if pattern in content:
            print(message)
        else:
            print(f"❌ Missing: {pattern}")
    print()


def check_vector_store_methods():
    """Verify VectorStore has community methods."""
    print("=" * 80)
    print("6. Checking VectorStore Methods")
    print("=" * 80)

    vector_store_path = Path("graphunified/storage/vector_store.py")
    content = vector_store_path.read_text()

    required_methods = [
        "index_communities",
        "search_communities",
    ]

    for method in required_methods:
        if f"async def {method}" in content:
            print(f"✅ VectorStore method exists: {method}")
        else:
            print(f"❌ VectorStore method missing: {method}")
    print()


def check_test_updates():
    """Verify test file is updated."""
    print("=" * 80)
    print("7. Checking Test Updates")
    print("=" * 80)

    test_path = Path("test_graphrag_global.py")
    content = test_path.read_text()

    checks = [
        ("from graphunified.storage.vector_store import VectorStore", "✅ VectorStore imported"),
        ("VectorDBConfig", "✅ VectorDBConfig imported"),
        ("vector_store = VectorStore.from_config", "✅ VectorStore initialized"),
        ("vector_store=vector_store", "✅ VectorStore passed to strategy"),
        ("SYNTHESIZED ANSWER", "✅ Test output updated for synthesis"),
    ]

    for pattern, message in checks:
        if pattern in content:
            print(message)
        else:
            print(f"❌ Missing: {pattern}")
    print()


def check_requirements():
    """Verify python-igraph is in requirements."""
    print("=" * 80)
    print("8. Checking Requirements")
    print("=" * 80)

    req_path = Path("requirements.txt")
    content = req_path.read_text()

    if "python-igraph" in content:
        print("✅ python-igraph in requirements.txt")
    else:
        print("❌ python-igraph missing from requirements.txt")
    print()


def main():
    """Run all verification checks."""
    print("\n")
    print("=" * 80)
    print("GraphRAG Global Enhancement Verification")
    print("=" * 80)
    print()

    check_imports()
    check_methods()
    check_leiden_algorithm()
    check_vector_store_integration()
    check_synthesis_in_retrieve()
    check_vector_store_methods()
    check_test_updates()
    check_requirements()

    print("=" * 80)
    print("Verification Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run: python test_graphrag_global.py")
    print("2. Verify:")
    print("   - Communities ranked by semantic similarity (not keyword count)")
    print("   - Single synthesized answer (not raw reports)")
    print("   - Answer cites multiple communities")
    print("   - Leiden algorithm logged")
    print("   - Retrieval time ~100-200ms (not <1ms)")
    print()


if __name__ == "__main__":
    main()
