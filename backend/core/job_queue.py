"""``JobQueue`` — deep wrapper around Azure Storage Queue for the async
image-job queue.

The queue carries only a small JSON pointer
``{"job_id": ..., "project_id": ...}`` — the real state lives in
``JobStore``. This separation lets a worker survive replica churn:
visibility-timeout re-delivery + JobStore lookup is enough to resume.

Public surface (per PRD § JobQueue + issue 002 AC):

  - ``enqueue(job_id, project_id)`` writes a pointer message to
    ``imagejobs`` with a 7-day TTL.
  - ``dequeue(visibility_timeout=90)`` returns one ``JobMessage`` or
    ``None`` if empty. 90s is the PRD-pinned visibility timeout.
  - ``complete(message)`` deletes a successfully processed message.
  - ``abandon(message)`` is the failure path. The 3rd failure routes
    to ``imagejobs-poison`` and deletes from ``imagejobs`` so a
    poisoned payload can never wedge the worker pool.

Auth uses ``DefaultAzureCredential`` (managed identity) — no
connection strings ever, per the codebase-wide rule and PRD AC.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient

from backend.core.config import settings

logger = logging.getLogger(__name__)


# Mirrors the KEDA / queue-service ``maxDequeueCount`` policy applied
# in ``infra/modules/storageAccount.bicep``. Both must agree — if they
# drift, the worker would either keep retrying forever (queue policy
# higher) or send to poison earlier than expected (queue policy lower).
MAX_DEQUEUE_COUNT = 3

VISIBILITY_TIMEOUT_SECONDS = 90
MESSAGE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

MAIN_QUEUE_NAME = "imagejobs"
POISON_QUEUE_NAME = "imagejobs-poison"


@dataclass
class JobMessage:
    """Decoded queue message exposing the fields the worker needs.

    ``raw`` holds the underlying SDK ``QueueMessage`` so ``complete``
    and ``abandon`` can pass it back to the SDK for delete/update by
    ``id`` + ``pop_receipt`` (which the SDK manages on the object).
    """

    job_id: str
    project_id: str
    dequeue_count: int
    raw: Any


class JobQueue:
    """Azure Storage Queue wrapper for image-pipeline jobs.

    Construct with ``main_client=`` / ``poison_client=`` for tests
    (mock or Azurite). Default construction uses managed identity to
    reach the configured storage account and ensures both queues exist.
    """

    def __init__(
        self,
        main_client: Optional[QueueClient] = None,
        poison_client: Optional[QueueClient] = None,
    ):
        if main_client is not None and poison_client is not None:
            self._main = main_client
            self._poison = poison_client
            return

        account_url = (
            f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.queue.core.windows.net/"
        )
        credential = DefaultAzureCredential()
        self._main = main_client or QueueClient(
            account_url=account_url,
            queue_name=MAIN_QUEUE_NAME,
            credential=credential,
        )
        self._poison = poison_client or QueueClient(
            account_url=account_url,
            queue_name=POISON_QUEUE_NAME,
            credential=credential,
        )
        for client, name in ((self._main, MAIN_QUEUE_NAME), (self._poison, POISON_QUEUE_NAME)):
            try:
                client.create_queue()
            except ResourceExistsError:
                pass
            except Exception as exc:  # noqa: BLE001 — log + continue; worker will retry on first send
                logger.warning("create_queue(%s) failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def enqueue(self, *, job_id: str, project_id: str) -> None:
        body = json.dumps({"job_id": job_id, "project_id": project_id})
        self._main.send_message(content=body, time_to_live=MESSAGE_TTL_SECONDS)
        logger.info("job.enqueued job_id=%s project_id=%s", job_id, project_id)

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def dequeue(
        self, visibility_timeout: int = VISIBILITY_TIMEOUT_SECONDS
    ) -> Optional[JobMessage]:
        messages = list(
            self._main.receive_messages(
                max_messages=1, visibility_timeout=visibility_timeout
            )
        )
        if not messages:
            return None
        raw = messages[0]
        payload = json.loads(raw.content)
        return JobMessage(
            job_id=payload["job_id"],
            project_id=payload["project_id"],
            dequeue_count=raw.dequeue_count,
            raw=raw,
        )

    def complete(self, message: JobMessage) -> None:
        """Delete a successfully processed message from the main queue."""
        self._main.delete_message(message.raw)

    def abandon(self, message: JobMessage) -> None:
        """Failure path. Re-deliver under the limit; poison at/over it.

        Note: ``message.dequeue_count`` reflects the count at the time
        ``dequeue`` returned the message — i.e. the attempt that just
        failed. So ``>= MAX_DEQUEUE_COUNT`` means: this was the Nth
        and final allowed attempt, do not redeliver.
        """
        if message.dequeue_count >= MAX_DEQUEUE_COUNT:
            body = json.dumps(
                {"job_id": message.job_id, "project_id": message.project_id}
            )
            self._poison.send_message(content=body, time_to_live=MESSAGE_TTL_SECONDS)
            self._main.delete_message(message.raw)
            logger.warning(
                "job.poisoned job_id=%s project_id=%s dequeue_count=%d",
                message.job_id,
                message.project_id,
                message.dequeue_count,
            )
            return
        # Re-deliver immediately — the visibility-timeout window has
        # already protected the previous attempt; abandoning means
        # "let another replica try now".
        self._main.update_message(
            message.raw, visibility_timeout=0, content=message.raw.content
        )
        logger.info(
            "job.abandoned job_id=%s dequeue_count=%d → redelivered",
            message.job_id,
            message.dequeue_count,
        )
