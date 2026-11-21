from unittest.mock import MagicMock

from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler


def _handler_returning(tags: list[str]) -> JiebaKeywordTableHandler:
    handler = JiebaKeywordTableHandler.__new__(JiebaKeywordTableHandler)
    handler._tfidf = MagicMock()
    handler._tfidf.extract_tags.return_value = tags
    return handler


def test_extract_keywords_passes_limit_to_tfidf():
    handler = _handler_returning(["dify"])

    handler.extract_keywords("some text", max_keywords_per_chunk=3)

    handler._tfidf.extract_tags.assert_called_once_with(sentence="some text", topK=3)


def test_extract_keywords_adds_subtokens_without_stopwords():
    handler = _handler_returning(["state-of-the-art", "dify"])

    keywords = handler.extract_keywords("text")

    assert keywords == {"state-of-the-art", "state", "art", "dify"}


def test_extract_keywords_keeps_single_word_tokens_unsplit():
    handler = _handler_returning(["hello_world", "检索增强"])

    assert handler.extract_keywords("text") == {"hello_world", "检索增强"}


def test_extract_keywords_with_real_jieba():
    keywords = JiebaKeywordTableHandler().extract_keywords("Dify 是一个开源的大语言模型应用开发平台", 5)

    assert keywords
    assert all(isinstance(keyword, str) for keyword in keywords)
