"""Console dataset endpoints must check the caller's dataset permission before acting on a dataset."""

import inspect
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from werkzeug.exceptions import Forbidden

import services
from controllers.console.datasets.datasets import DatasetAutoDisableLogApi, DatasetErrorDocs
from controllers.console.datasets.datasets_document import (
    DocumentPauseApi,
    DocumentPipelineExecutionLogApi,
    DocumentRecoverApi,
    DocumentRetryApi,
)

_DATASETS = "controllers.console.datasets.datasets"
_DOCUMENTS = "controllers.console.datasets.datasets_document"

_DENIED = services.errors.account.NoPermissionError("You do not have permission to access this dataset.")


def _call(module: str, handler: Callable, *args, json: dict | None = None):
    app = Flask(__name__)
    dataset = MagicMock(id="ds-1", tenant_id="tenant-other")
    with (
        app.test_request_context("/", method="POST", json=json or {}),
        patch(f"{module}.current_account_with_tenant", return_value=(MagicMock(), "tenant-1")),
        patch(f"{module}.DatasetService.get_dataset", return_value=dataset),
        patch(f"{module}.DatasetService.check_dataset_permission", side_effect=_DENIED),
        patch(f"{module}.DocumentService") as document_service,
    ):
        document_service.get_document.return_value = MagicMock(id="doc-1", tenant_id="tenant-other")
        document_service.check_archived.return_value = False
        try:
            return inspect.unwrap(handler)(*args)
        finally:
            _call.document_service = document_service  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("module", "resource", "method", "args", "json"),
    [
        (_DATASETS, DatasetErrorDocs, "get", ("ds-1",), None),
        (_DATASETS, DatasetAutoDisableLogApi, "get", ("ds-1",), None),
        (_DOCUMENTS, DocumentPipelineExecutionLogApi, "get", ("ds-1", "doc-1"), None),
        (_DOCUMENTS, DocumentPauseApi, "patch", ("ds-1", "doc-1"), None),
        (_DOCUMENTS, DocumentRecoverApi, "patch", ("ds-1", "doc-1"), None),
        (_DOCUMENTS, DocumentRetryApi, "post", ("ds-1",), {"document_ids": ["doc-1"]}),
    ],
    ids=["error-docs", "auto-disable-logs", "pipeline-log", "pause", "recover", "retry"],
)
def test_endpoint_rejects_caller_without_dataset_permission(module, resource, method, args, json):
    with pytest.raises(Forbidden):
        _call(module, getattr(resource, method), resource(), *args, json=json)

    document_service = _call.document_service  # type: ignore[attr-defined]
    document_service.pause_document.assert_not_called()
    document_service.recover_document.assert_not_called()
    document_service.retry_document.assert_not_called()
    document_service.get_error_documents_by_dataset_id.assert_not_called()
