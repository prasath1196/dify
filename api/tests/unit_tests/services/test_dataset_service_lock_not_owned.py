import types
from unittest.mock import Mock, create_autospec

import pytest
from redis.exceptions import LockNotOwnedError

from models.account import Account
from models.dataset import Dataset, Document
from services.dataset_service import DocumentService, SegmentService


class FakeLock:
    """Lock that always fails on enter with LockNotOwnedError."""

    def __enter__(self):
        raise LockNotOwnedError("simulated")

    def __exit__(self, exc_type, exc, tb):
        # Normal contextmanager signature; return False so exceptions propagate
        return False


@pytest.fixture
def fake_current_user(monkeypatch):
    user = create_autospec(Account, instance=True)
    user.id = "user-1"
    user.current_tenant_id = "tenant-1"
    monkeypatch.setattr("services.dataset_service.current_user", user)
    return user


@pytest.fixture
def fake_features(monkeypatch):
    """Features.billing.enabled == False to skip quota logic."""
    features = types.SimpleNamespace(
        billing=types.SimpleNamespace(enabled=False, subscription=types.SimpleNamespace(plan="ENTERPRISE")),
        documents_upload_quota=types.SimpleNamespace(limit=10_000, size=0),
    )
    monkeypatch.setattr(
        "services.dataset_service.FeatureService.get_features",
        lambda tenant_id: features,
    )
    return features


@pytest.fixture
def fake_lock(monkeypatch):
    """Patch redis_client.lock to always raise LockNotOwnedError on enter."""

    def _fake_lock(name, timeout=None, *args, **kwargs):
        return FakeLock()

    # DatasetService imports redis_client directly from extensions.ext_redis
    monkeypatch.setattr("services.dataset_service.redis_client.lock", _fake_lock)


# ---------------------------------------------------------------------------
# 1. Knowledge Pipeline document creation (save_document_with_dataset_id)
# ---------------------------------------------------------------------------


def test_save_document_with_dataset_id_ignores_lock_not_owned(
    monkeypatch,
    fake_current_user,
    fake_features,
    fake_lock,
):
    # Arrange
    dataset = create_autospec(Dataset, instance=True)
    dataset.id = "ds-1"
    dataset.tenant_id = fake_current_user.current_tenant_id
    dataset.data_source_type = "upload_file"
    dataset.indexing_technique = "high_quality"  # so we skip re-initialization branch

    # Minimal knowledge_config stub that satisfies pre-lock code
    info_list = types.SimpleNamespace(data_source_type="upload_file")
    data_source = types.SimpleNamespace(info_list=info_list)
    knowledge_config = types.SimpleNamespace(
        doc_form="qa_model",
        original_document_id=None,  # go into "new document" branch
        data_source=data_source,
        indexing_technique="high_quality",
        embedding_model=None,
        embedding_model_provider=None,
        retrieval_model=None,
        process_rule=None,
        duplicate=False,
        doc_language="en",
    )

    account = fake_current_user

    # Avoid touching real doc_form logic
    monkeypatch.setattr("services.dataset_service.DatasetService.check_doc_form", lambda *a, **k: None)
    # Avoid real DB interactions
    monkeypatch.setattr("services.dataset_service.db", Mock())

    # Act: this would hit the redis lock, whose __enter__ raises LockNotOwnedError.
    # Our implementation should catch it and still return (documents, batch).
    documents, batch = DocumentService.save_document_with_dataset_id(
        dataset=dataset,
        knowledge_config=knowledge_config,
        account=account,
    )

    # Assert
    # We mainly care that:
    # - No exception is raised
    # - The function returns a sensible tuple
    assert isinstance(documents, list)
    assert isinstance(batch, str)


# ---------------------------------------------------------------------------
# 2. Single-segment creation (add_segment)
# ---------------------------------------------------------------------------


def test_add_segment_ignores_lock_not_owned(
    monkeypatch,
    fake_current_user,
    fake_lock,
):
    # Arrange
    dataset = create_autospec(Dataset, instance=True)
    dataset.id = "ds-1"
    dataset.tenant_id = fake_current_user.current_tenant_id
    dataset.indexing_technique = "economy"  # skip embedding/token calculation branch

    document = create_autospec(Document, instance=True)
    document.id = "doc-1"
    document.dataset_id = dataset.id
    document.word_count = 0
    document.doc_form = "qa_model"

    # Minimal args required by add_segment
    args = {
        "content": "question text",
        "answer": "answer text",
        "keywords": ["k1", "k2"],
    }

    # Avoid real DB operations
    db_mock = Mock()
    db_mock.session = Mock()
    monkeypatch.setattr("services.dataset_service.db", db_mock)
    monkeypatch.setattr("services.dataset_service.VectorService", Mock())

    # Act
    result = SegmentService.create_segment(args=args, document=document, dataset=dataset)

    # Assert
    # Under LockNotOwnedError except, add_segment should swallow the error and return None.
    assert result is None


# ---------------------------------------------------------------------------
# 3. Multi-segment creation (multi_create_segment)
# ---------------------------------------------------------------------------


def test_multi_create_segment_ignores_lock_not_owned(
    monkeypatch,
    fake_current_user,
    fake_lock,
):
    # Arrange
    dataset = create_autospec(Dataset, instance=True)
    dataset.id = "ds-1"
    dataset.tenant_id = fake_current_user.current_tenant_id
    dataset.indexing_technique = "economy"  # again, skip high_quality path

    document = create_autospec(Document, instance=True)
    document.id = "doc-1"
    document.dataset_id = dataset.id
    document.word_count = 0
    document.doc_form = "qa_model"

    segments = [{"content": "question text", "answer": "answer text", "keywords": ["k1"]}]

    # Avoid real DB operations
    db_mock = Mock()
    db_mock.session = Mock()
    monkeypatch.setattr("services.dataset_service.db", db_mock)
    vector_service = Mock()
    monkeypatch.setattr("services.dataset_service.VectorService", vector_service)

    # Act
    result = SegmentService.multi_create_segment(segments=segments, document=document, dataset=dataset)

    # Assert
    # Under LockNotOwnedError, multi_create_segment swallows the error, writes nothing and returns None.
    assert result is None
    db_mock.session.add.assert_not_called()
    db_mock.session.commit.assert_not_called()
    vector_service.create_segments_vector.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Lock expiring before release, after the work is done
# ---------------------------------------------------------------------------


class ExpiredOnReleaseLock:
    """Lock that is acquired but has expired by the time it is released."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        raise LockNotOwnedError("lock expired before release")


@pytest.fixture
def expired_on_release_lock(monkeypatch):
    monkeypatch.setattr(
        "services.dataset_service.redis_client.lock", lambda name, timeout=None, *args, **kwargs: ExpiredOnReleaseLock()
    )


def _economy_qa_dataset_and_document(fake_current_user):
    dataset = create_autospec(Dataset, instance=True)
    dataset.id = "ds-1"
    dataset.tenant_id = fake_current_user.current_tenant_id
    dataset.indexing_technique = "economy"

    document = create_autospec(Document, instance=True)
    document.id = "doc-1"
    document.dataset_id = dataset.id
    document.word_count = 0
    document.doc_form = "qa_model"
    return dataset, document


def test_create_segment_returns_segment_when_lock_expires_before_release(
    monkeypatch, fake_current_user, expired_on_release_lock
):
    dataset, document = _economy_qa_dataset_and_document(fake_current_user)
    created_segment = Mock()
    db_mock = Mock()
    db_mock.session.query.return_value.where.return_value.first.return_value = created_segment
    db_mock.session.query.return_value.where.return_value.scalar.return_value = 0
    monkeypatch.setattr("services.dataset_service.db", db_mock)
    monkeypatch.setattr("services.dataset_service.VectorService", Mock())

    result = SegmentService.create_segment(
        args={"content": "question text", "answer": "answer text"}, document=document, dataset=dataset
    )

    assert result is created_segment
    db_mock.session.commit.assert_called()


def test_multi_create_segment_returns_segments_when_lock_expires_before_release(
    monkeypatch, fake_current_user, expired_on_release_lock
):
    dataset, document = _economy_qa_dataset_and_document(fake_current_user)
    db_mock = Mock()
    db_mock.session.query.return_value.where.return_value.scalar.return_value = 0
    monkeypatch.setattr("services.dataset_service.db", db_mock)
    monkeypatch.setattr("services.dataset_service.VectorService", Mock())

    result = SegmentService.multi_create_segment(
        segments=[{"content": "q1", "answer": "a1"}, {"content": "q2", "answer": "a2"}],
        document=document,
        dataset=dataset,
    )

    assert result is not None
    assert [segment.content for segment in result] == ["q1", "q2"]
    db_mock.session.commit.assert_called()


class RecordingLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.held = False
        return False


def test_create_segment_commits_new_segment_while_holding_the_position_lock(monkeypatch, fake_current_user):
    dataset, document = _economy_qa_dataset_and_document(fake_current_user)
    lock = RecordingLock()
    monkeypatch.setattr("services.dataset_service.redis_client.lock", lambda name, timeout=None: lock)
    lock_held_at_commit: list[bool] = []
    db_mock = Mock()
    db_mock.session.query.return_value.where.return_value.scalar.return_value = 3
    db_mock.session.commit.side_effect = lambda: lock_held_at_commit.append(lock.held)
    monkeypatch.setattr("services.dataset_service.db", db_mock)
    monkeypatch.setattr("services.dataset_service.VectorService", Mock())

    SegmentService.create_segment(
        args={"content": "question text", "answer": "answer text"}, document=document, dataset=dataset
    )

    added_segment = db_mock.session.add.call_args_list[0].args[0]
    assert added_segment.position == 4
    assert lock_held_at_commit[0] is True
