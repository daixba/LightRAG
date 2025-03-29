import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, final
import numpy as np

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger
import configparser

from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    AWSV4SignerAuth,
    NotFoundError,
    RequestError,
)
import boto3

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class OpenSearchVectorDBStorage(BaseVectorStorage):
    """OpenSearch Serverless vector storage implementation."""

    def __post_init__(self):
        try:
            vector_db_config = self.global_config.get("vector_db_storage_cls_kwargs", {})
            self.cosine_better_than_threshold = vector_db_config.get("cosine_better_than_threshold", 0.3)

            self.region = os.environ.get("AWS_REGION", config.get("opensearch", "region_name", fallback="us-east-1"))
            self.host = os.environ.get(
                "OPENSEARCH_ENDPOINT", config.get("opensearch", "endpoint", fallback=None)
            ).strip("https://")
            self.service = "aoss"

            # OpenSearch Serverless specific configuration
            self.collection_name = self.namespace.lower()  # OpenSearch requires lowercase
            if not self.host:
                raise ValueError("OpenSearch host must be specified")

            # Initialize AWS credentials
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, self.region, service=self.service)

            # Initialize OpenSearch client
            self._client = OpenSearch(
                hosts=[{"host": self.host, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300,
            )

            # Create index with vector search configuration if it doesn't exist
            if not self._client.indices.exists(self.collection_name):
                index_body = {
                    "mappings": {
                        "properties": {
                            "vector": {
                                "type": "knn_vector",
                                "dimension": self.embedding_func.embedding_dim,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "l2",
                                    "engine": "faiss",
                                    "parameters": {
                                        "ef_construction": 128,
                                        "m": 16,
                                    },
                                },
                            },
                            "content": {"type": "text"},
                            "metadata": {"type": "object"},
                        }
                    },
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 128,
                        }
                    },
                }
                self._client.indices.create(
                    index=self.collection_name,
                    body=index_body,
                )

            # Set batch size for processing
            self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

        except Exception as e:
            logger.error(f"OpenSearch initialization failed: {str(e)}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.collection_name}")
        if not data:
            return

        try:
            # Process documents in batches
            for i in range(0, len(data), self._max_batch_size):
                batch = dict(list(data.items())[i : i + self._max_batch_size])

                # Get documents and metadata for the batch
                documents = [v["content"] for v in batch.values()]
                metadatas = [
                    {k: v for k, v in item.items() if k in self.meta_fields} or {"_default": "true"}
                    for item in batch.values()
                ]

                # Generate embeddings for the batch
                embeddings = await self.embedding_func(documents)

                # Prepare bulk operation
                bulk_data = []

                for doc_id, doc, metadata, embedding in zip(batch.keys(), documents, metadatas, embeddings):
                    # Index action
                    # , "_id": doc_id is not supported
                    bulk_data.append({"index": {"_index": self.collection_name}})
                    # Document data
                    bulk_data.append(
                        {
                            "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                            "content": doc,
                            "metadata": metadata,
                        }
                    )

                # Execute bulk operation
                if bulk_data:
                    response = self._client.bulk(body=bulk_data)
                    if response.get("errors", False):
                        logger.error(f"Bulk indexing errors: {response}")
                        raise Exception("Bulk indexing failed")

            return list(data.keys())

        except Exception as e:
            logger.error(f"Error during OpenSearch upsert: {str(e)}")
            raise

    async def query(self, query: str, top_k: int, ids: list[str] | None = None) -> list[dict[str, Any]]:
        try:
            # Generate query embedding
            embedding = await self.embedding_func([query])
            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            # Prepare search query
            search_query = {
                "size": top_k * 2,  # Request more results to allow for filtering
                "query": {"knn": {"vector": {"vector": vector[0], "k": top_k * 2}}},
            }

            if ids:
                search_query["query"] = {
                    "bool": {
                        "must": [
                            {"ids": {"values": ids}},
                            {"knn": {"vector": {"vector": vector[0], "k": top_k * 2}}},
                        ]
                    }
                }

            # Execute search
            response = self._client.search(index=self.collection_name, body=search_query)

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                # OpenSearch returns scores between 0 and 1, where 1 is most similar
                # Convert to distance (0 = identical, 1 = orthogonal)
                distance = 1 - hit["_score"]

                if distance >= self.cosine_better_than_threshold:
                    results.append(
                        {
                            "id": hit["_id"],
                            "distance": distance,
                            "content": hit["_source"]["content"],
                            **hit["_source"]["metadata"],
                        }
                    )

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error during OpenSearch query: {str(e)}")
            raise

    async def index_done_callback(self) -> None:
        # OpenSearch handles index updates automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its ID."""
        try:
            logger.info(f"Deleting entity with ID {entity_name} from {self.collection_name}")
            self._client.delete(index=self.collection_name, id=entity_name, refresh=True)
        except NotFoundError:
            logger.warning(f"Entity {entity_name} not found in {self.collection_name}")
        except Exception as e:
            logger.error(f"Error during entity deletion: {str(e)}")
            raise

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete an entity and its relations by ID.
        In vector DB context, this is equivalent to delete_entity.

        Args:
            entity_name: The ID of the entity to delete
        """
        await self.delete_entity(entity_name)

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        try:
            logger.info(f"Deleting {len(ids)} vectors from {self.collection_name}")

            # Prepare bulk delete operation
            bulk_data = []
            for doc_id in ids:
                bulk_data.append({"delete": {"_index": self.collection_name, "_id": doc_id}})

            if bulk_data:
                response = self._client.bulk(body=bulk_data, refresh=True)
                if response.get("errors", False):
                    logger.error(f"Bulk deletion errors: {response}")
                    raise Exception("Bulk deletion failed")

            logger.debug(f"Successfully deleted {len(ids)} vectors from {self.collection_name}")

        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.collection_name}: {e}")
            raise

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        try:
            # Use wildcard query to match IDs with prefix
            search_query = {
                "query": {"wildcard": {"_id": f"{prefix}*"}},
                "size": 10000,  # Adjust based on your needs
            }

            response = self._client.search(index=self.collection_name, body=search_query)

            results = []
            for hit in response["hits"]["hits"]:
                results.append(
                    {
                        "id": hit["_id"],
                        "content": hit["_source"]["content"],
                        "vector": hit["_source"]["vector"],
                        **hit["_source"]["metadata"],
                    }
                )

            logger.debug(f"Found {len(results)} records with prefix '{prefix}'")
            return results

        except Exception as e:
            logger.error(f"Error during prefix search in OpenSearch: {str(e)}")
            raise

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            try:
                result = self._client.get(index=self.collection_name, id=id)
                return {
                    "id": result["_id"],
                    "vector": result["_source"]["vector"],
                    "content": result["_source"]["content"],
                    **result["_source"]["metadata"],
                }
            except NotFoundError:
                return None

        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Use mget to retrieve multiple documents
            body = {"ids": ids}
            response = self._client.mget(body=body, index=self.collection_name)

            results = []
            for doc in response["docs"]:
                if doc.get("found", False):
                    results.append(
                        {
                            "id": doc["_id"],
                            "vector": doc["_source"]["vector"],
                            "content": doc["_source"]["content"],
                            **doc["_source"]["metadata"],
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []
