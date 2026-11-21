"""Security tests for console file uploads.

The FileApi handlers are called through inspect.unwrap so the setup/login/billing decorators are skipped
and the real request validation runs; FileService is mocked. The FileService tests exercise the real
filename and extension checks, which run before any storage or database access.
"""

import inspect
import io
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from sqlalchemy.orm import sessionmaker
from werkzeug.exceptions import Forbidden

from configs import dify_config
from controllers.common.errors import (
    BlockedFileExtensionError,
    FilenameNotExistsError,
    FileTooLargeError,
    NoFileUploadedError,
    TooManyFilesError,
    UnsupportedFileTypeError,
)
from controllers.console.files import FileApi
from services.errors.file import BlockedFileExtensionError as ServiceBlockedFileExtensionError
from services.errors.file import FileTooLargeError as ServiceFileTooLargeError
from services.errors.file import UnsupportedFileTypeError as ServiceUnsupportedFileTypeError
from services.file_service import FileService

_MODULE = "controllers.console.files"


def _post(files: dict, *, source: str | None = None, is_dataset_editor: bool = True):
    app = Flask(__name__)
    user = MagicMock(is_dataset_editor=is_dataset_editor)
    data: dict = dict(files)
    if source is not None:
        data["source"] = source
    with (
        app.test_request_context("/files/upload", method="POST", data=data, content_type="multipart/form-data"),
        patch(f"{_MODULE}.current_account_with_tenant", return_value=(user, "tenant-1")),
        patch(f"{_MODULE}.db"),
        patch(f"{_MODULE}.FileService") as file_service_cls,
    ):
        file_service_cls.return_value.upload_file.return_value = "uploaded-file"
        result = inspect.unwrap(FileApi.post)(FileApi())
    return result, file_service_cls.return_value.upload_file


def _file(content: bytes = b"hello", filename: str = "notes.txt") -> tuple[io.BytesIO, str]:
    return (io.BytesIO(content), filename)


class TestFileApiUploadValidation:
    def test_rejects_request_without_file(self):
        with pytest.raises(NoFileUploadedError):
            _post({})

    def test_rejects_multiple_files(self):
        with pytest.raises(TooManyFilesError):
            _post({"file": _file(), "other": _file(filename="other.txt")})

    def test_rejects_empty_filename(self):
        with pytest.raises(FilenameNotExistsError):
            _post({"file": _file(filename="")})

    def test_dataset_upload_requires_dataset_editor(self):
        with pytest.raises(Forbidden):
            _post({"file": _file(filename="doc.pdf")}, source="datasets", is_dataset_editor=False)

    def test_general_upload_does_not_require_dataset_editor(self):
        result, upload_file = _post({"file": _file()}, is_dataset_editor=False)

        assert result == ("uploaded-file", 201)
        assert upload_file.call_args.kwargs["source"] is None

    def test_unknown_source_is_treated_as_general_upload(self):
        _, upload_file = _post({"file": _file()}, source="somewhere-else")

        assert upload_file.call_args.kwargs["source"] is None

    def test_passes_file_content_and_name_to_service(self):
        _, upload_file = _post({"file": _file(b"payload", "report.txt")}, source="datasets")

        kwargs = upload_file.call_args.kwargs
        assert (kwargs["filename"], kwargs["content"], kwargs["source"]) == ("report.txt", b"payload", "datasets")


class TestFileApiServiceErrorMapping:
    @pytest.mark.parametrize(
        ("service_error", "api_error"),
        [
            (ServiceFileTooLargeError("too large"), FileTooLargeError),
            (ServiceUnsupportedFileTypeError(), UnsupportedFileTypeError),
            (ServiceBlockedFileExtensionError("blocked"), BlockedFileExtensionError),
        ],
    )
    def test_service_errors_become_api_errors(self, service_error: Exception, api_error: type[Exception]):
        app = Flask(__name__)
        with (
            app.test_request_context(
                "/files/upload", method="POST", data={"file": _file()}, content_type="multipart/form-data"
            ),
            patch(f"{_MODULE}.current_account_with_tenant", return_value=(MagicMock(), "tenant-1")),
            patch(f"{_MODULE}.db"),
            patch(f"{_MODULE}.FileService") as file_service_cls,
        ):
            file_service_cls.return_value.upload_file.side_effect = service_error
            with pytest.raises(api_error):
                inspect.unwrap(FileApi.post)(FileApi())


class TestFileApiUploadConfiguration:
    def test_returns_configured_limits(self):
        app = Flask(__name__)
        with app.test_request_context("/files/upload"):
            body, status = inspect.unwrap(FileApi.get)(FileApi())

        assert status == 200
        assert body["file_size_limit"] == dify_config.UPLOAD_FILE_SIZE_LIMIT
        assert body["batch_count_limit"] == dify_config.UPLOAD_FILE_BATCH_LIMIT
        assert body["image_file_size_limit"] == dify_config.UPLOAD_IMAGE_FILE_SIZE_LIMIT


class TestFileServiceFilenameSecurity:
    @staticmethod
    def _upload(filename: str, source: str | None = None):
        service = FileService(MagicMock(spec=sessionmaker))
        return service.upload_file(
            filename=filename, content=b"data", mimetype="text/plain", user=MagicMock(), source=source
        )

    @pytest.mark.parametrize(
        "filename", ["../etc/passwd", "..\\windows\\system.ini", "dir/file.txt", "a:b.txt", "evil\x00.txt"]
    )
    def test_rejects_path_separators_and_unsafe_characters(self, filename: str):
        with pytest.raises(ValueError, match="invalid characters"):
            self._upload(filename)

    def test_rejects_blacklisted_extension(self):
        with patch.object(type(dify_config), "UPLOAD_FILE_EXTENSION_BLACKLIST", {"exe"}):
            with pytest.raises(ServiceBlockedFileExtensionError):
                self._upload("installer.EXE")

    def test_dataset_upload_requires_document_extension(self):
        with pytest.raises(ServiceUnsupportedFileTypeError):
            self._upload("image.png", source="datasets")
