"""Cosmos DB CRUD operations for staging projects."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from azure.cosmos import ContainerProxy, exceptions
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

from backend.core.config import settings

logger = logging.getLogger(__name__)


class StagingStorageService:
    """Manages StagingProject documents in Cosmos DB."""

    def __init__(self, container: Optional[ContainerProxy] = None):
        if container is not None:
            self.container = container
            return
        credential = DefaultAzureCredential()
        client = CosmosClient(url=settings.AZURE_COSMOS_DB_ENDPOINT, credential=credential)
        database = client.get_database_client(settings.AZURE_COSMOS_DB_ID)
        self.container = database.create_container_if_not_exists(
            id=settings.STAGING_COSMOS_CONTAINER_ID,
            partition_key={"paths": ["/id"], "kind": "Hash"},
        )

    def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in project_data:
            project_data["id"] = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        project_data["created_at"] = now
        project_data["updated_at"] = now
        project_data["doc_type"] = "staging_project"
        return self.container.create_item(body=project_data)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.container.read_item(item=project_id, partition_key=project_id)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.get_project(project_id)
        if not existing:
            raise ValueError(f"Staging project not found: {project_id}")
        existing.update(updates)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.container.replace_item(item=project_id, body=existing)

    def list_projects(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = "SELECT * FROM c WHERE c.doc_type = 'staging_project' ORDER BY c.created_at DESC OFFSET @offset LIMIT @limit"
        params = [{"name": "@offset", "value": offset}, {"name": "@limit", "value": limit}]
        return list(self.container.query_items(query=query, parameters=params, enable_cross_partition_query=True))

    def count_projects(self) -> int:
        query = "SELECT VALUE COUNT(1) FROM c WHERE c.doc_type = 'staging_project'"
        results = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        return results[0] if results else 0

    def delete_project(self, project_id: str) -> bool:
        try:
            self.container.delete_item(item=project_id, partition_key=project_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
