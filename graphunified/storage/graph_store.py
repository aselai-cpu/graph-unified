"""Graph database storage for entity-relationship networks."""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import networkx as nx
from networkx.algorithms import community

from graphunified.config.models import Entity, Relationship
from graphunified.config.settings import StorageConfig
from graphunified.exceptions import StorageError
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class GraphStore:
    """NetworkX-based graph storage for entity-relationship networks.

    Supports:
    - Directed and undirected graph construction
    - Graph traversal operations (neighbors, paths, subgraphs)
    - Community detection helpers (Leiden, Louvain)
    - Serialization to GraphML and pickle formats
    - Metadata storage for nodes and edges
    """

    def __init__(
        self,
        root_dir: Path,
        directed: bool = True,
        graph_format: str = "pickle",
    ):
        """Initialize graph store.

        Args:
            root_dir: Root directory for graph storage
            directed: Whether to use directed graph (default: True)
            graph_format: Serialization format ('pickle' or 'graphml')
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.directed = directed
        self.graph_format = graph_format

        # Initialize empty graph
        self.graph: nx.DiGraph | nx.Graph = nx.DiGraph() if directed else nx.Graph()

        # File paths
        self.graph_file = self.root_dir / f"graph.{graph_format}"

    @classmethod
    def from_config(cls, config: StorageConfig, root_dir: Path) -> "GraphStore":
        """Create graph store from configuration.

        Args:
            config: Storage configuration
            root_dir: Root directory for storage

        Returns:
            GraphStore instance
        """
        return cls(
            root_dir=root_dir,
            directed=True,  # Default to directed graph
            graph_format=config.graph_format,
        )

    async def build_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> None:
        """Build graph from entities and relationships.

        Args:
            entities: List of entities to add as nodes
            relationships: List of relationships to add as edges

        Raises:
            StorageError: If graph construction fails
        """
        try:
            # Clear existing graph
            self.graph.clear()

            # Add entities as nodes
            for entity in entities:
                self.graph.add_node(
                    str(entity.id),
                    name=entity.name,
                    type=entity.type.value,
                    description=entity.description or "",
                    metadata=entity.metadata,
                )

            logger.info(f"Added {len(entities)} entities as nodes")

            # Add relationships as edges
            edge_count = 0
            for rel in relationships:
                # Skip if nodes don't exist (defensive)
                if not self.graph.has_node(str(rel.source_entity_id)) or not self.graph.has_node(
                    str(rel.target_entity_id)
                ):
                    logger.warning(
                        f"Skipping relationship {rel.id}: missing source or target node"
                    )
                    continue

                self.graph.add_edge(
                    str(rel.source_entity_id),
                    str(rel.target_entity_id),
                    id=str(rel.id),
                    type=rel.type.value,
                    description=rel.description or "",
                    weight=rel.weight,
                    metadata=rel.metadata,
                )
                edge_count += 1

            logger.info(f"Added {edge_count} relationships as edges")

        except Exception as e:
            raise StorageError(f"Failed to build graph: {e}")

    async def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to existing graph.

        Args:
            entities: Entities to add
        """
        for entity in entities:
            self.graph.add_node(
                str(entity.id),
                name=entity.name,
                type=entity.type.value,
                description=entity.description or "",
                metadata=entity.metadata,
            )

        logger.info(f"Added {len(entities)} entities to graph")

    async def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to existing graph.

        Args:
            relationships: Relationships to add
        """
        edge_count = 0
        for rel in relationships:
            if not self.graph.has_node(str(rel.source_entity_id)) or not self.graph.has_node(
                str(rel.target_entity_id)
            ):
                logger.warning(
                    f"Skipping relationship {rel.id}: missing source or target node"
                )
                continue

            self.graph.add_edge(
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                id=str(rel.id),
                type=rel.type.value,
                description=rel.description or "",
                weight=rel.weight,
                metadata=rel.metadata,
            )
            edge_count += 1

        logger.info(f"Added {edge_count} relationships to graph")

    async def get_neighbors(
        self,
        entity_id: UUID,
        max_hops: int = 1,
        entity_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get neighbors of an entity within max hops.

        Args:
            entity_id: Entity ID to start from
            max_hops: Maximum number of hops (default: 1)
            entity_types: Optional filter by entity types

        Returns:
            List of (node_id, node_attributes) tuples

        Raises:
            StorageError: If entity not found
        """
        node_id = str(entity_id)

        if not self.graph.has_node(node_id):
            raise StorageError(f"Entity {entity_id} not found in graph")

        try:
            # Get all nodes within max_hops using BFS
            neighbors = set()
            visited = {node_id}
            current_level = {node_id}

            for _ in range(max_hops):
                next_level = set()
                for node in current_level:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in visited:
                            neighbors.add(neighbor)
                            visited.add(neighbor)
                            next_level.add(neighbor)
                current_level = next_level

            # Filter by entity types if specified
            if entity_types:
                neighbors = {
                    n
                    for n in neighbors
                    if self.graph.nodes[n].get("type") in entity_types
                }

            # Return neighbors with attributes
            result = [(n, dict(self.graph.nodes[n])) for n in neighbors]

            logger.debug(f"Found {len(result)} neighbors for entity {entity_id}")
            return result

        except Exception as e:
            raise StorageError(f"Failed to get neighbors: {e}")

    async def get_subgraph(
        self,
        entity_ids: List[UUID],
        max_hops: int = 1,
    ) -> nx.Graph | nx.DiGraph:
        """Extract subgraph around specified entities.

        Args:
            entity_ids: Entity IDs to center subgraph around
            max_hops: Maximum distance from seed entities

        Returns:
            NetworkX subgraph

        Raises:
            StorageError: If extraction fails
        """
        try:
            # Get all neighbors within max_hops
            subgraph_nodes = set()
            for entity_id in entity_ids:
                node_id = str(entity_id)
                if self.graph.has_node(node_id):
                    subgraph_nodes.add(node_id)

                    # BFS to get neighbors
                    visited = {node_id}
                    current_level = {node_id}

                    for _ in range(max_hops):
                        next_level = set()
                        for node in current_level:
                            for neighbor in self.graph.neighbors(node):
                                if neighbor not in visited:
                                    subgraph_nodes.add(neighbor)
                                    visited.add(neighbor)
                                    next_level.add(neighbor)
                        current_level = next_level

            # Extract subgraph
            subgraph = self.graph.subgraph(subgraph_nodes).copy()

            logger.info(
                f"Extracted subgraph with {subgraph.number_of_nodes()} nodes, "
                f"{subgraph.number_of_edges()} edges"
            )

            return subgraph

        except Exception as e:
            raise StorageError(f"Failed to extract subgraph: {e}")

    async def shortest_path(
        self,
        source_id: UUID,
        target_id: UUID,
        weight: Optional[str] = "weight",
    ) -> List[str]:
        """Find shortest path between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            weight: Edge attribute to use as weight (default: 'weight')

        Returns:
            List of node IDs in path

        Raises:
            StorageError: If path cannot be found
        """
        source = str(source_id)
        target = str(target_id)

        if not self.graph.has_node(source):
            raise StorageError(f"Source entity {source_id} not found")
        if not self.graph.has_node(target):
            raise StorageError(f"Target entity {target_id} not found")

        try:
            path = nx.shortest_path(self.graph, source, target, weight=weight)
            logger.debug(f"Found path of length {len(path) - 1} from {source} to {target}")
            return path

        except nx.NetworkXNoPath:
            raise StorageError(f"No path found between {source_id} and {target_id}")
        except Exception as e:
            raise StorageError(f"Failed to find shortest path: {e}")

    async def get_connected_components(self) -> List[Set[str]]:
        """Get connected components in the graph.

        Returns:
            List of sets, each containing node IDs in a component
        """
        try:
            if self.directed:
                # For directed graphs, use weakly connected components
                components = list(nx.weakly_connected_components(self.graph))
            else:
                components = list(nx.connected_components(self.graph))

            logger.info(f"Found {len(components)} connected components")
            return components

        except Exception as e:
            raise StorageError(f"Failed to get connected components: {e}")

    async def detect_communities_louvain(
        self,
        resolution: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """Detect communities using Louvain algorithm.

        Args:
            resolution: Resolution parameter (default: 1.0)
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping node_id -> community_id

        Raises:
            StorageError: If community detection fails
        """
        try:
            # Louvain requires undirected graph
            if self.directed:
                graph = self.graph.to_undirected()
            else:
                graph = self.graph

            # Run Louvain algorithm
            communities = community.louvain_communities(
                graph,
                resolution=resolution,
                seed=seed,
            )

            # Convert to node -> community_id mapping
            node_to_community = {}
            for community_id, nodes in enumerate(communities):
                for node in nodes:
                    node_to_community[node] = community_id

            logger.info(f"Detected {len(communities)} communities using Louvain")
            return node_to_community

        except Exception as e:
            raise StorageError(f"Failed to detect communities: {e}")

    async def detect_communities_leiden(
        self,
        resolution: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """Detect communities using Leiden algorithm (requires igraph).

        Args:
            resolution: Resolution parameter (default: 1.0)
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping node_id -> community_id

        Raises:
            StorageError: If community detection fails
        """
        try:
            import igraph as ig

            # Convert NetworkX graph to igraph
            if self.directed:
                graph = self.graph.to_undirected()
            else:
                graph = self.graph

            # Get edges with weights
            edges = []
            weights = []
            node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
            idx_to_node = {idx: node for node, idx in node_to_idx.items()}

            for u, v, data in graph.edges(data=True):
                edges.append((node_to_idx[u], node_to_idx[v]))
                weights.append(data.get("weight", 1.0))

            # Create igraph Graph
            ig_graph = ig.Graph(n=len(node_to_idx), edges=edges, directed=False)
            ig_graph.es["weight"] = weights

            # Run Leiden algorithm
            if seed is not None:
                ig.set_random_number_generator(ig.RNG(seed))

            communities = ig_graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution_parameter=resolution,
            )

            # Convert back to NetworkX node IDs
            node_to_community = {}
            for community_id, nodes in enumerate(communities):
                for node_idx in nodes:
                    node_id = idx_to_node[node_idx]
                    node_to_community[node_id] = community_id

            logger.info(f"Detected {len(communities)} communities using Leiden")
            return node_to_community

        except ImportError:
            raise StorageError(
                "python-igraph not installed. Install with: pip install python-igraph"
            )
        except Exception as e:
            raise StorageError(f"Failed to detect communities with Leiden: {e}")

    async def get_node_attributes(self, entity_id: UUID) -> Dict[str, Any]:
        """Get attributes for a specific node.

        Args:
            entity_id: Entity ID

        Returns:
            Dictionary of node attributes

        Raises:
            StorageError: If node not found
        """
        node_id = str(entity_id)

        if not self.graph.has_node(node_id):
            raise StorageError(f"Entity {entity_id} not found in graph")

        return dict(self.graph.nodes[node_id])

    async def get_edge_attributes(
        self, source_id: UUID, target_id: UUID
    ) -> Dict[str, Any]:
        """Get attributes for a specific edge.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            Dictionary of edge attributes

        Raises:
            StorageError: If edge not found
        """
        source = str(source_id)
        target = str(target_id)

        if not self.graph.has_edge(source, target):
            raise StorageError(f"Edge from {source_id} to {target_id} not found")

        return dict(self.graph[source][target])

    async def save(self) -> None:
        """Save graph to disk.

        Raises:
            StorageError: If save fails
        """
        try:
            if self.graph_format == "pickle":
                await asyncio.to_thread(self._save_pickle)
            elif self.graph_format == "graphml":
                await asyncio.to_thread(self._save_graphml)
            else:
                raise ValueError(f"Unsupported graph format: {self.graph_format}")

            logger.info(
                f"Saved graph with {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges to {self.graph_file}"
            )

        except Exception as e:
            raise StorageError(f"Failed to save graph: {e}")

    def _save_pickle(self) -> None:
        """Save graph as pickle (internal)."""
        with open(self.graph_file, "wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_graphml(self) -> None:
        """Save graph as GraphML (internal)."""
        nx.write_graphml(self.graph, self.graph_file)

    async def load(self) -> None:
        """Load graph from disk.

        Raises:
            StorageError: If load fails
        """
        if not self.graph_file.exists():
            logger.warning(f"Graph file {self.graph_file} does not exist")
            return

        try:
            if self.graph_format == "pickle":
                await asyncio.to_thread(self._load_pickle)
            elif self.graph_format == "graphml":
                await asyncio.to_thread(self._load_graphml)
            else:
                raise ValueError(f"Unsupported graph format: {self.graph_format}")

            logger.info(
                f"Loaded graph with {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges from {self.graph_file}"
            )

        except Exception as e:
            raise StorageError(f"Failed to load graph: {e}")

    def _load_pickle(self) -> None:
        """Load graph from pickle (internal)."""
        with open(self.graph_file, "rb") as f:
            self.graph = pickle.load(f)

    def _load_graphml(self) -> None:
        """Load graph from GraphML (internal)."""
        self.graph = nx.read_graphml(self.graph_file)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "directed": self.directed,
        }

        # Add density (only if graph has nodes)
        if stats["num_nodes"] > 0:
            stats["density"] = nx.density(self.graph)

        # Add average degree
        if stats["num_nodes"] > 0:
            degrees = [d for _, d in self.graph.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees)
        else:
            stats["avg_degree"] = 0.0

        # Add connectivity info
        if stats["num_nodes"] > 0:
            if self.directed:
                stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(
                    self.graph
                )
            else:
                stats["num_connected_components"] = nx.number_connected_components(
                    self.graph
                )

        return stats
