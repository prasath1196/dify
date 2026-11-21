from unittest.mock import MagicMock, patch

import pytest

from services.metadata_service import MetadataService

_MODULE = "services.metadata_service"


def test_lock_check_acquires_lock_atomically():
    with patch(f"{_MODULE}.redis_client") as redis_client:
        redis_client.set.return_value = True

        MetadataService.knowledge_base_metadata_lock_check("dataset-1", None)

    redis_client.set.assert_called_once_with("dataset_metadata_lock_dataset-1", 1, ex=3600, nx=True)


def test_lock_check_raises_when_lock_is_held():
    with patch(f"{_MODULE}.redis_client") as redis_client:
        redis_client.set.return_value = None

        with pytest.raises(ValueError, match="Another knowledge base metadata operation is running"):
            MetadataService.knowledge_base_metadata_lock_check("dataset-1", None)


def test_delete_metadata_does_not_release_a_lock_held_by_another_operation():
    with patch(f"{_MODULE}.redis_client") as redis_client, patch(f"{_MODULE}.db"):
        redis_client.set.return_value = None

        with pytest.raises(ValueError, match="Another knowledge base metadata operation is running"):
            MetadataService.delete_metadata("dataset-1", "metadata-1")

    redis_client.delete.assert_not_called()


def test_delete_metadata_rolls_back_and_releases_lock_on_failure():
    with patch(f"{_MODULE}.redis_client") as redis_client, patch(f"{_MODULE}.db") as db:
        redis_client.set.return_value = True
        db.session.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Metadata not found"):
            MetadataService.delete_metadata("dataset-1", "metadata-1")

    db.session.rollback.assert_called_once()
    redis_client.delete.assert_called_once_with("dataset_metadata_lock_dataset-1")


def test_enable_built_in_field_propagates_lock_contention():
    dataset = MagicMock(id="dataset-1", built_in_field_enabled=False)
    with patch(f"{_MODULE}.redis_client") as redis_client, patch(f"{_MODULE}.db"):
        redis_client.set.return_value = None

        with pytest.raises(ValueError, match="Another knowledge base metadata operation is running"):
            MetadataService.enable_built_in_field(dataset)

    redis_client.delete.assert_not_called()
