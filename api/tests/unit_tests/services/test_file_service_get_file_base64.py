import base64
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker
from werkzeug.exceptions import NotFound

from services.file_service import FileService


def _make_service(upload_file: object | None) -> tuple[FileService, MagicMock]:
    session = MagicMock()
    session.__enter__.return_value = session
    session.query.return_value.where.return_value.first.return_value = upload_file
    session_maker = MagicMock(spec=sessionmaker, return_value=session)
    return FileService(session_maker), session


def test_get_file_base64_encodes_blob_and_closes_session():
    upload_file = MagicMock(key="upload_files/tenant/file.png")
    service, session = _make_service(upload_file)

    with patch("services.file_service.storage.load_once", return_value=b"png-bytes") as load_once:
        result = service.get_file_base64("file-1")

    load_once.assert_called_once_with("upload_files/tenant/file.png")
    assert result == base64.b64encode(b"png-bytes").decode()
    session.__exit__.assert_called_once()


def test_get_file_base64_raises_not_found_and_closes_session():
    service, session = _make_service(None)

    with pytest.raises(NotFound):
        service.get_file_base64("missing")

    session.__exit__.assert_called_once()
