import pytest

from core.rag.extractor.extract_processor import ExtractProcessor


@pytest.mark.parametrize(
    ("url", "headers", "expected"),
    [
        ("https://example.com/report.pdf", {}, ".pdf"),
        ("https://example.com/report.pdf?token=abc", {}, ".pdf"),
        ("https://example.com/download", {"Content-Type": "text/html; charset=utf-8"}, ".html"),
        (
            "https://example.com/download",
            {"Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
            ".docx",
        ),
        (
            "https://example.com/download",
            {"Content-Type": "application/octet-stream", "Content-Disposition": 'attachment; filename="notes.md"'},
            ".md",
        ),
        ("https://example.com/download", {"Content-Type": "application/octet-stream"}, ""),
        ("https://example.com/download", {}, ""),
    ],
)
def test_download_suffix(url: str, headers: dict[str, str], expected: str):
    assert ExtractProcessor._download_suffix(url, headers) == expected
