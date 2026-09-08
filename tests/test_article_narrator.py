"""Unit tests for the Web Link Article Narrator service."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.services.ai.article_reader import (
    extract_article_content,
    is_safe_public_url,
    summarize_article_with_ai,
)


class TestSafePublicUrlValidation(unittest.TestCase):
    """Test SSRF protections on input URLs."""

    def test_valid_public_urls(self) -> None:
        # Using domain that resolves to public IP
        safe, reason = is_safe_public_url("https://example.com/news/article")
        self.assertTrue(safe)
        self.assertEqual("", reason)

    def test_blocks_loopback_and_localhost(self) -> None:
        safe, reason = is_safe_public_url("http://127.0.0.1:8080/admin")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

        safe, reason = is_safe_public_url("http://localhost:3000/")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_cloud_metadata(self) -> None:
        safe, reason = is_safe_public_url("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_non_http_schemes(self) -> None:
        safe, reason = is_safe_public_url("ftp://example.com/file.txt")
        self.assertFalse(safe)
        self.assertIn("only http and https", reason.lower())

        safe, reason = is_safe_public_url("file:///etc/passwd")
        self.assertFalse(safe)
        self.assertIn("only http and https", reason.lower())

    def test_blocks_empty_url(self) -> None:
        safe, reason = is_safe_public_url("")
        self.assertFalse(safe)


class TestArticleHtmlExtraction(unittest.TestCase):
    """Test HTML parsing and clean article text extraction."""

    def test_extracts_title_and_clean_paragraphs(self) -> None:
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking News: Major Tech Discovery</title>
            <meta property="og:title" content="Breaking News: Major Tech Discovery" />
            <script>var x = 100; alert('spam');</script>
            <style>body { background: red; }</style>
        </head>
        <body>
            <nav><a href="/">Home</a> <a href="/news">News</a></nav>
            <header><h1>Header Banner</h1></header>
            <main>
                <article>
                    <h1>Breaking News: Major Tech Discovery</h1>
                    <p>Scientists and software engineers have announced a breakthrough in speech synthesis today.</p>
                    <p>The new technology allows computers to read complex multilingual documents with zero latency.</p>
                </article>
            </main>
            <footer>Copyright 2026 Example Corp</footer>
        </body>
        </html>
        """
        title, body = extract_article_content(sample_html)
        self.assertEqual("Breaking News: Major Tech Discovery", title)
        self.assertIn("Scientists and software engineers", body)
        self.assertIn("breakthrough in speech synthesis", body)
        self.assertNotIn("alert('spam')", body)
        self.assertNotIn("Copyright 2026", body)
        self.assertNotIn("Home", body)

    def test_og_description_fallback_when_body_sparse(self) -> None:
        sparse_html = """
        <html>
        <head>
            <meta property="og:title" content="Short Update" />
            <meta property="og:description" content="This is an informative summary from social meta tags." />
        </head>
        <body><div>Hi</div></body>
        </html>
        """
        title, body = extract_article_content(sparse_html)
        self.assertEqual("Short Update", title)
        self.assertIn("informative summary from social meta tags", body)


class TestArticleSummarization(unittest.TestCase):
    """Test AI summarization logic."""

    def test_fallback_excerpt_when_client_none(self) -> None:
        body = (
            "Paragraph one describing important background details.\n\n"
            "Paragraph two describing the latest developments today.\n\n"
            "Paragraph three with concluding remarks."
        )
        summary = summarize_article_with_ai("Sample Title", body, gemini_client=None)
        self.assertIn("Paragraph one", summary)
        self.assertIn("Paragraph two", summary)

    def test_ai_summarization_with_mock_client(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "• Summary point 1\n• Summary point 2"
        mock_client.models.generate_content.return_value = mock_resp

        summary = summarize_article_with_ai("Test Title", "Long article body text", gemini_client=mock_client)
        self.assertIn("Summary point 1", summary)


if __name__ == "__main__":
    unittest.main()
