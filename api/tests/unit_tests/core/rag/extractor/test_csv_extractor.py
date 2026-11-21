from pathlib import Path

import pytest

from core.rag.extractor.csv_extractor import CSVExtractor


def _write_csv(tmp_path: Path, content: str, encoding: str = "utf-8") -> str:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding=encoding)
    return str(path)


def test_extract_one_document_per_row(tmp_path: Path):
    path = _write_csv(tmp_path, "name,city\nAlice,Paris\nBob,Berlin\n")

    docs = CSVExtractor(path).extract()

    assert [doc.page_content for doc in docs] == ["name: Alice;city: Paris", "name: Bob;city: Berlin"]
    assert [doc.metadata["row"] for doc in docs] == [0, 1]


def test_extract_skips_empty_cells_instead_of_writing_nan(tmp_path: Path):
    path = _write_csv(tmp_path, "name,email,age\nAlice,,30\nBob,bob@example.com,\n")

    docs = CSVExtractor(path).extract()

    assert [doc.page_content for doc in docs] == ["name: Alice;age: 30", "name: Bob;email: bob@example.com"]


def test_extract_keeps_values_as_written(tmp_path: Path):
    path = _write_csv(tmp_path, "code,amount,flag\n007,1.50,NA\n")

    docs = CSVExtractor(path).extract()

    assert docs[0].page_content == "code: 007;amount: 1.50;flag: NA"


def test_extract_uses_source_column(tmp_path: Path):
    path = _write_csv(tmp_path, "url,title\nhttps://example.com,Home\n")

    docs = CSVExtractor(path, source_column="url").extract()

    assert docs[0].metadata["source"] == "https://example.com"


def test_extract_rejects_missing_source_column(tmp_path: Path):
    path = _write_csv(tmp_path, "title\nHome\n")

    with pytest.raises(ValueError, match="Source column 'url' not found"):
        CSVExtractor(path, source_column="url").extract()
