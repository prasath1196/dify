"""Tests for the console dataset, hit testing and external knowledge API endpoints.

The handlers are called through inspect.unwrap so the setup/login/billing decorators are bypassed and
only the endpoint logic runs; services and the provider manager are mocked.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from werkzeug.exceptions import Forbidden, NotFound

from controllers.console.datasets.datasets import DatasetApi, DatasetListApi
from controllers.console.datasets.external import ExternalApiTemplateListApi
from controllers.console.datasets.hit_testing import HitTestingApi
from models.account import Account

_MODULE = "controllers.console.datasets.datasets"


def _dataset(dataset_id: str = "dataset-1", name: str = "Dataset", **overrides) -> SimpleNamespace:
    values = {
        "id": dataset_id,
        "name": name,
        "description": "",
        "permission": "only_me",
        "indexing_technique": "economy",
        "embedding_model": None,
        "embedding_model_provider": None,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def app() -> Flask:
    return Flask(__name__)


@pytest.fixture
def current_user():
    user = MagicMock(id="user-1")
    with patch(f"{_MODULE}.current_account_with_tenant", return_value=(user, "tenant-1")):
        yield user


@pytest.fixture
def no_embedding_models():
    with patch(f"{_MODULE}.ProviderManager") as provider_manager:
        provider_manager.return_value.get_configurations.return_value.get_models.return_value = []
        yield provider_manager


def _list_datasets(app: Flask, query: str):
    with app.test_request_context(f"/datasets{query}"):
        return inspect.unwrap(DatasetListApi.get)(DatasetListApi())


class TestDatasetListApi:
    def test_returns_paginated_datasets(self, app, current_user, no_embedding_models):
        datasets = [_dataset(f"dataset-{i}", f"Dataset {i}") for i in range(3)]

        with patch(f"{_MODULE}.DatasetService.get_datasets", return_value=(datasets, 3)) as get_datasets:
            body, status = _list_datasets(app, "?page=1&limit=20")

        assert status == 200
        assert [item["id"] for item in body["data"]] == ["dataset-0", "dataset-1", "dataset-2"]
        assert (body["total"], body["page"], body["limit"], body["has_more"]) == (3, 1, 20, False)
        get_datasets.assert_called_once_with(1, 20, "tenant-1", current_user, None, [], False)

    def test_passes_keyword_search_to_service(self, app, current_user, no_embedding_models):
        with patch(f"{_MODULE}.DatasetService.get_datasets", return_value=([], 0)) as get_datasets:
            body, status = _list_datasets(app, "?keyword=manual")

        assert status == 200
        assert body["data"] == []
        assert get_datasets.call_args.args[4] == "manual"

    def test_reports_has_more_when_page_is_full(self, app, current_user, no_embedding_models):
        datasets = [_dataset(f"dataset-{i}") for i in range(2)]

        with patch(f"{_MODULE}.DatasetService.get_datasets", return_value=(datasets, 5)) as get_datasets:
            body, _ = _list_datasets(app, "?page=2&limit=2")

        assert (body["page"], body["limit"], body["total"], body["has_more"]) == (2, 2, 5, True)
        assert get_datasets.call_args.args[:2] == (2, 2)


class TestDatasetApiGet:
    def test_returns_dataset_detail(self, app, current_user, no_embedding_models):
        dataset = _dataset("dataset-1", "Product docs")

        with (
            patch(f"{_MODULE}.DatasetService.get_dataset", return_value=dataset),
            patch(f"{_MODULE}.DatasetService.check_dataset_permission") as check_permission,
            app.test_request_context("/datasets/dataset-1"),
        ):
            body, status = inspect.unwrap(DatasetApi.get)(DatasetApi(), "dataset-1")

        assert status == 200
        assert (body["id"], body["name"], body["embedding_available"]) == ("dataset-1", "Product docs", True)
        check_permission.assert_called_once_with(dataset, current_user)

    def test_missing_dataset_raises_not_found(self, app, current_user):
        with (
            patch(f"{_MODULE}.DatasetService.get_dataset", return_value=None),
            app.test_request_context("/datasets/missing"),
        ):
            with pytest.raises(NotFound):
                inspect.unwrap(DatasetApi.get)(DatasetApi(), "missing")


class TestDatasetListApiCreate:
    def test_creates_dataset(self, app, current_user):
        current_user.is_dataset_editor = True
        created = _dataset("dataset-new", "New dataset")

        with (
            patch(f"{_MODULE}.DatasetService.create_empty_dataset", return_value=created) as create_dataset,
            app.test_request_context("/datasets", method="POST", json={"name": "New dataset"}),
        ):
            body, status = inspect.unwrap(DatasetListApi.post)(DatasetListApi())

        assert status == 201
        assert (body["id"], body["name"]) == ("dataset-new", "New dataset")
        assert create_dataset.call_args.kwargs["name"] == "New dataset"
        assert create_dataset.call_args.kwargs["tenant_id"] == "tenant-1"

    def test_non_editor_is_forbidden(self, app, current_user):
        current_user.is_dataset_editor = False

        with (
            patch(f"{_MODULE}.DatasetService.create_empty_dataset") as create_dataset,
            app.test_request_context("/datasets", method="POST", json={"name": "New dataset"}),
        ):
            with pytest.raises(Forbidden):
                inspect.unwrap(DatasetListApi.post)(DatasetListApi())

        create_dataset.assert_not_called()


class TestHitTestingApi:
    def test_returns_query_and_records(self, app):
        account = MagicMock(spec=Account)
        dataset = _dataset("dataset-1")
        with (
            patch("controllers.console.datasets.hit_testing_base.current_user", account),
            patch("controllers.console.datasets.hit_testing_base.DatasetService.get_dataset", return_value=dataset),
            patch("controllers.console.datasets.hit_testing_base.DatasetService.check_dataset_permission"),
            patch(
                "controllers.console.datasets.hit_testing_base.HitTestingService.retrieve",
                return_value={"query": {"content": "refund policy"}, "records": []},
            ) as retrieve,
            app.test_request_context("/datasets/dataset-1/hit-testing", method="POST", json={"query": "refund policy"}),
        ):
            body = inspect.unwrap(HitTestingApi.post)(HitTestingApi(), "dataset-1")

        assert body == {"query": {"content": "refund policy"}, "records": []}
        assert retrieve.call_args.kwargs["query"] == "refund policy"
        assert retrieve.call_args.kwargs["dataset"] is dataset


class TestExternalApiTemplateListApi:
    def test_returns_paginated_external_apis(self, app):
        external_apis = [MagicMock(to_dict=MagicMock(return_value={"id": f"api-{i}"})) for i in range(2)]
        with (
            patch(
                "controllers.console.datasets.external.current_account_with_tenant",
                return_value=(MagicMock(), "tenant-1"),
            ),
            patch(
                "controllers.console.datasets.external.ExternalDatasetService.get_external_knowledge_apis",
                return_value=(external_apis, 2),
            ) as get_apis,
            app.test_request_context("/datasets/external-knowledge-api?page=1&limit=20&keyword=wiki"),
        ):
            body, status = inspect.unwrap(ExternalApiTemplateListApi.get)(ExternalApiTemplateListApi())

        assert status == 200
        assert body["data"] == [{"id": "api-0"}, {"id": "api-1"}]
        assert (body["total"], body["has_more"]) == (2, False)
        get_apis.assert_called_once_with(1, 20, "tenant-1", "wiki")
