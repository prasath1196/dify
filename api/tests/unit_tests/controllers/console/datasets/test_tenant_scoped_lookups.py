"""Cross-tenant access checks for console dataset endpoints."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from werkzeug.exceptions import Forbidden, NotFound

import services
from controllers.console.datasets.datasets_segments import DatasetDocumentSegmentBatchImportApi
from controllers.console.datasets.rag_pipeline.rag_pipeline import CustomizedPipelineTemplateApi

_SEGMENTS = "controllers.console.datasets.datasets_segments"


def _batch_import(user: MagicMock, *, permission_error: Exception | None = None, upload_file=None):
    app = Flask(__name__)
    with (
        app.test_request_context(
            "/datasets/ds-1/documents/doc-1/segments/batch_import",
            method="POST",
            json={"upload_file_id": "file-1"},
        ),
        patch(f"{_SEGMENTS}.current_account_with_tenant", return_value=(user, "tenant-1")),
        patch(f"{_SEGMENTS}.DatasetService.get_dataset", return_value=MagicMock(id="ds-1")),
        patch(f"{_SEGMENTS}.DocumentService.get_document", return_value=MagicMock(id="doc-1")),
        patch(f"{_SEGMENTS}.DatasetService.check_dataset_permission", side_effect=permission_error) as check,
        patch(f"{_SEGMENTS}.db") as db,
        patch(f"{_SEGMENTS}.redis_client"),
        patch(f"{_SEGMENTS}.batch_create_segment_to_index_task") as task,
    ):
        db.session.query.return_value.where.return_value.first.return_value = upload_file
        try:
            result = inspect.unwrap(DatasetDocumentSegmentBatchImportApi.post)(
                DatasetDocumentSegmentBatchImportApi(), "ds-1", "doc-1"
            )
        finally:
            lookup_conditions = db.session.query.return_value.where.call_args
    return result, check, task, lookup_conditions


class TestSegmentBatchImportAuthorization:
    def test_rejects_users_without_dataset_permission(self):
        user = MagicMock(is_dataset_editor=True)

        with pytest.raises(Forbidden):
            _batch_import(user, permission_error=services.errors.account.NoPermissionError("no access"))

    def test_rejects_non_editors(self):
        user = MagicMock(is_dataset_editor=False)

        with pytest.raises(Forbidden):
            _batch_import(user)

    def test_missing_upload_file_is_not_found(self):
        user = MagicMock(is_dataset_editor=True)

        with pytest.raises(NotFound):
            _batch_import(user, upload_file=None)

    def test_permitted_editor_queues_import(self):
        user = MagicMock(is_dataset_editor=True)
        upload_file = MagicMock(name="upload_file")
        upload_file.name = "segments.csv"

        result, check, task, conditions = _batch_import(user, upload_file=upload_file)

        check.assert_called_once()
        task.delay.assert_called_once()
        assert result[1] == 200
        rendered = " ".join(str(condition) for condition in conditions.args)
        assert "upload_files.tenant_id" in rendered


class TestCustomizedPipelineTemplateExport:
    def test_template_lookup_is_scoped_to_current_tenant(self):
        app = Flask(__name__)
        session = MagicMock()
        session.query.return_value.where.return_value.first.return_value = MagicMock(yaml_content="graph: {}")

        with (
            app.test_request_context("/rag/pipeline/customized/templates/t-1", method="POST"),
            patch(
                "controllers.console.datasets.rag_pipeline.rag_pipeline.current_account_with_tenant",
                return_value=(MagicMock(), "tenant-1"),
            ),
            patch("controllers.console.datasets.rag_pipeline.rag_pipeline.db"),
            patch("controllers.console.datasets.rag_pipeline.rag_pipeline.Session") as session_cls,
        ):
            session_cls.return_value.__enter__.return_value = session
            body, status = inspect.unwrap(CustomizedPipelineTemplateApi.post)(CustomizedPipelineTemplateApi(), "t-1")

        assert (body, status) == ({"data": "graph: {}"}, 200)
        rendered = " ".join(str(condition) for condition in session.query.return_value.where.call_args.args)
        assert "pipeline_customized_templates.tenant_id" in rendered
