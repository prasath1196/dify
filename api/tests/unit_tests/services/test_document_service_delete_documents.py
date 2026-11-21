from unittest.mock import MagicMock, patch

from services.dataset_service import DocumentService


def test_delete_documents_only_touches_documents_in_the_dataset():
    dataset = MagicMock(id="ds-1", doc_form="text_model")
    own_document = MagicMock(
        id="doc-own", data_source_type="upload_file", data_source_info_dict={"upload_file_id": "f1"}
    )

    with (
        patch("services.dataset_service.db") as db,
        patch("services.dataset_service.batch_clean_document_task") as clean_task,
    ):
        db.session.scalars.return_value.all.return_value = [own_document]
        DocumentService.delete_documents(dataset, ["doc-own", "doc-other-tenant"])

    statement = db.session.scalars.call_args.args[0]
    assert "documents.dataset_id" in str(statement)
    clean_task.delay.assert_called_once_with(["doc-own"], "ds-1", "text_model", ["f1"])
    db.session.delete.assert_called_once_with(own_document)


def test_delete_documents_with_no_matching_documents_does_nothing():
    dataset = MagicMock(id="ds-1", doc_form="text_model")

    with (
        patch("services.dataset_service.db") as db,
        patch("services.dataset_service.batch_clean_document_task") as clean_task,
    ):
        db.session.scalars.return_value.all.return_value = []
        DocumentService.delete_documents(dataset, ["doc-other-tenant"])

    clean_task.delay.assert_not_called()
    db.session.delete.assert_not_called()
