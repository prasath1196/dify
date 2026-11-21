import pytest

from core.rag.cleaner.clean_processor import CleanProcessor


def _rules(*enabled_rule_ids: str) -> dict:
    return {
        "rules": {
            "pre_processing_rules": [
                {"id": "remove_extra_spaces", "enabled": "remove_extra_spaces" in enabled_rule_ids},
                {"id": "remove_urls_emails", "enabled": "remove_urls_emails" in enabled_rule_ids},
            ]
        }
    }


@pytest.mark.parametrize("text", ["naïve café", "¿Qué tal?", "¾ cup of flour", "Ångström, Øre, ß"])
def test_clean_preserves_non_ascii_letters_and_punctuation(text: str):
    assert CleanProcessor.clean(text, {}) == text


def test_clean_removes_control_characters_and_fffe():
    assert CleanProcessor.clean("a\x00b\x08c\x0bd\x1fe\x7ff￾g", {}) == "abcdefg"


def test_clean_keeps_tab_and_newlines():
    assert CleanProcessor.clean("a\tb\nc\r\nd", {}) == "a\tb\nc\r\nd"


def test_clean_neutralizes_special_token_delimiters():
    assert CleanProcessor.clean("<|endoftext|>", {}) == "<endoftext>"


def test_clean_remove_extra_spaces_collapses_blank_lines_and_spaces():
    text = "first\n\n\n\nsecond   word　　end"

    assert CleanProcessor.clean(text, _rules("remove_extra_spaces")) == "first\n\nsecond word end"


def test_clean_leaves_spacing_when_rule_disabled():
    text = "first\n\n\n\nsecond   word"

    assert CleanProcessor.clean(text, _rules()) == text


def test_clean_remove_urls_emails_keeps_markdown_image_urls():
    text = "mail me@example.com or see https://example.com/page ![alt](https://img.example.com/a.png)"

    assert CleanProcessor.clean(text, _rules("remove_urls_emails")) == (
        "mail  or see  ![image](https://img.example.com/a.png)"
    )
