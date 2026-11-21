from pathlib import Path
from unittest.mock import patch

from docx import Document as DocxDocument

from core.rag.extractor.entity.datasource_type import DatasourceType
from core.rag.extractor.entity.extract_setting import ExtractSetting
from core.rag.extractor.extract_processor import ExtractProcessor


def test_extract_docx_from_file_path_without_upload_file(tmp_path: Path):
    docx_path = tmp_path / "remote.docx"
    document = DocxDocument()
    document.add_paragraph("Hello from a downloaded document")
    document.save(str(docx_path))
    extract_setting = ExtractSetting(datasource_type=DatasourceType.FILE, document_model="text_model")

    with patch("core.rag.extractor.extract_processor.dify_config.ETL_TYPE", "dify"):
        docs = ExtractProcessor.extract(extract_setting=extract_setting, file_path=str(docx_path))

    assert "Hello from a downloaded document" in docs[0].page_content
