import os
from abc import ABC
from dataclasses import dataclass
from typing import Any, Union, final
import boto3
from botocore.exceptions import ClientError
import configparser

from lightrag.base import (
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.utils import (
    logger,
)
from ..namespace import NameSpace, is_namespace
from .shared_storage import (
    get_namespace_data,
    get_storage_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class DynamoDBKVStorage(BaseKVStorage):
    def _get_table_name(self) -> str:
        return os.environ.get("DYNAMO_KV_TABLE_NAME", config.get("dynamo", "kv_table_name", fallback=None))

    def __post_init__(self):
        # Initialize DynamoDB client
        REGION = os.environ.get("AWS_REGION", config.get("dynamo", "region_name", fallback="us-east-1"))
        self.table_name = self._get_table_name()
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=REGION,
        )
        self.table = self.dynamodb.Table(self.table_name)

    async def initialize(self):
        """Initialize storage data"""
        pass

    async def index_done_callback(self) -> None:
        # DynamoDB handles persistence automatically
        pass

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage"""
        raise NotImplementedError("get_all is not implemented for DynamoDB")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Return an item by id from DynamoDB"""
        try:
            response = self.table.get_item(Key={"id": id, "namespace": self.namespace})
            item = response.get("Item")
            return item if item else None
        except ClientError as e:
            logger.error(f"Error getting item from DynamoDB: {str(e)}")
            raise

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Return multiple items by their ids from DynamoDB using batch get"""
        if not ids:
            return []

        try:
            # DynamoDB batch_get_item has a limit of 100 items per request
            results = []
            for i in range(0, len(ids), 100):
                batch_ids = ids[i : i + 100]
                response = self.dynamodb.batch_get_item(
                    RequestItems={
                        self.table_name: {
                            "Keys": [{"id": id, "namespace": self.namespace} for id in batch_ids],
                            "ConsistentRead": True,
                        }
                    }
                )

                # Create a mapping of id to data for ordering
                items_map = {item["id"]: item for item in response.get("Responses", {}).get(self.table_name, [])}

                # Maintain input order and handle missing items
                results.extend(items_map.get(id) for id in batch_ids)

            return [r for r in results if r is not None]
        except ClientError as e:
            logger.error(f"Error batch getting items from DynamoDB: {str(e)}")
            raise

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        # For Response Cache
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            res = {}
            _id = f"{mode}_{id}"
            v = await self.get_by_id(_id)
            if v:
                res[id] = v
                return res
            return
        else:
            return None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that don't exist in DynamoDB"""
        if not keys:
            return set()

        # Convert set to list for get_by_ids
        keys_list = list(keys)

        try:
            # Use get_by_ids to efficiently fetch existing items
            existing_items = await self.get_by_ids(keys_list)

            # Create set of existing keys (items that were found)
            existing_keys = {keys_list[i] for i, item in enumerate(existing_items) if item is not None}

            # Return keys that weren't found
            return keys - existing_keys

        except ClientError as e:
            logger.error(f"Error filtering keys from DynamoDB: {str(e)}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert items into DynamoDB table"""
        if not data:
            return

        items = []
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for mode, p_items in data.items():
                # use sub items.
                for k, v in p_items.items():
                    id = f"{mode}_{k}"
                    items.append(
                        {
                            "id": id,
                            "namespace": self.namespace,
                            **v,
                        }
                    )
        else:
            for k, v in data.items():
                items.append(
                    {
                        "id": k,
                        "namespace": self.namespace,
                        **v,
                    }
                )

        total = len(items)
        logger.info(f"Inserting {total} records to {self.namespace}")

        try:
            # Batch write items to DynamoDB
            for i in range(0, total, 25):
                with self.table.batch_writer() as batch:
                    for item in items[i : i + 25]:
                        batch.put_item(Item=item)

            logger.info(f"Process {os.getpid()} writing {len(items)} records to {self.namespace}")
        except ClientError as e:
            logger.error(f"Error writing data to DynamoDB: {str(e)}")
            raise

    async def delete(self, ids: list[str]) -> None:
        """Delete multiple items from DynamoDB by their ids"""
        if not ids:
            return

        try:
            # DynamoDB batch_writer has a limit of 25 items per batch
            for i in range(0, len(ids), 25):
                batch_ids = ids[i : i + 25]
                with self.table.batch_writer() as batch:
                    for doc_id in batch_ids:
                        batch.delete_item(
                            Key={
                                "namespace": self.namespace,
                                "id": doc_id,
                            }
                        )
            logger.info(f"Deleted {len(ids)} items from namespace {self.namespace}")
        except ClientError as e:
            logger.error(f"Error deleting items from DynamoDB: {str(e)}")
            raise


@final
@dataclass
class DynamoDBDocStatusStorage(DynamoDBKVStorage, DocStatusStorage):
    """DynamoDB implementation of document status storage"""

    def _get_table_name(self) -> str:
        return os.environ.get(
            "DYNAMO_DOCSTATUS_TABLE_NAME", config.get("dynamo", "docstatus_table_name", fallback=None)
        )

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        raise NotImplementedError("Not implemented yet")

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status using the status-id GSI"""
        try:
            # Query the status-id GSI using expression attribute names to handle reserved keywords
            response = self.table.query(
                IndexName="status-id-index",  # Name of the GSI
                KeyConditionExpression="#status = :status_val",
                ExpressionAttributeNames={
                    "#status": "status"  # Use doc_status as the actual attribute name
                },
                ExpressionAttributeValues={":status_val": status.value},
            )

            # Convert DynamoDB items to DocProcessingStatus objects
            result = {}
            for item in response.get("Items", []):
                result[item["id"]] = DocProcessingStatus(
                    content=item.get("content", ""),
                    content_summary=item.get("content_summary", ""),
                    content_length=item.get("content_length", 0),
                    file_path=item.get("file_path", ""),
                    status=DocStatus(item.get("status", status.value)),
                    created_at=item.get("created_at", ""),
                    updated_at=item.get("updated_at", ""),
                    chunks_count=item.get("chunks_count"),
                    error=item.get("error"),
                    metadata=item.get("metadata", {}),
                )

            return result

        except ClientError as e:
            logger.error(f"Error querying documents by status from DynamoDB: {str(e)}")
            raise

    async def drop(self) -> None:
        """Drop the storage"""
        raise NotImplementedError("Not implemented yet")
