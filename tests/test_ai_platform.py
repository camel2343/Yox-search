import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from donersearch import ai_platform as aimod
from donersearch import db as dbmod


class AIPlatformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.conn = dbmod.open_db(self.tmp.name)
        dbmod.ensure_schema(self.conn)
        dbmod.upsert_document(
            self.conn,
            "https://example.com/page",
            "Example",
            "Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            56,
            aimod.utc_now(),
            "tr",
            "hash-v1",
            None,
        )
        self.conn.commit()

    def tearDown(self):
        try:
            self.conn.close()
        finally:
            if os.path.exists(self.tmp.name):
                os.unlink(self.tmp.name)

    def test_chunk_text_overlaps_and_preserves_order(self):
        words = " ".join(f"w{i}" for i in range(30))
        chunks = aimod.chunk_text(words, max_tokens=10, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("w0"))
        self.assertIn("w8 w9", chunks[0])
        self.assertTrue(chunks[1].startswith("w8"))

    def test_sync_document_pipeline_reuses_same_version(self):
        doc_id = dbmod.get_doc_id_by_url(self.conn, "https://example.com/page")
        first = aimod.sync_document_pipeline(
            self.conn,
            doc_id=doc_id,
            url="https://example.com/page",
            title="Example",
            raw_html="<html>v1</html>",
            content="Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            language="tr",
            content_hash="hash-v1",
        )
        second = aimod.sync_document_pipeline(
            self.conn,
            doc_id=doc_id,
            url="https://example.com/page",
            title="Example",
            raw_html="<html>v1</html>",
            content="Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            language="tr",
            content_hash="hash-v1",
        )
        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        cur = self.conn.execute("SELECT COUNT(*) FROM raw_snapshots")
        self.assertEqual(cur.fetchone()[0], 1)
        cur = self.conn.execute("SELECT COUNT(*) FROM document_chunks WHERE active=1")
        self.assertGreater(cur.fetchone()[0], 0)

    def test_prepare_dataset_writes_manifest(self):
        doc_id = dbmod.get_doc_id_by_url(self.conn, "https://example.com/page")
        aimod.sync_document_pipeline(
            self.conn,
            doc_id=doc_id,
            url="https://example.com/page",
            title="Example",
            raw_html="<html>v1</html>",
            content="Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            language="tr",
            content_hash="hash-v1",
        )
        manifest = aimod.prepare_dataset(self.conn, dataset_version="dataset_test")
        self.assertEqual(manifest["dataset_version"], "dataset_test")
        self.assertGreaterEqual(manifest["document_count"], 1)
        self.assertGreaterEqual(manifest["chunk_count"], 1)

    def test_build_embeddings_skips_unchanged_documents(self):
        doc_id = dbmod.get_doc_id_by_url(self.conn, "https://example.com/page")
        aimod.sync_document_pipeline(
            self.conn,
            doc_id=doc_id,
            url="https://example.com/page",
            title="Example",
            raw_html="<html>embed</html>",
            content="Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            language="tr",
            content_hash="hash-v1",
        )
        with mock.patch("donersearch.ai_platform.embmod.embed_batch", return_value=[np.ones(768, dtype=np.float32)]):
            first = aimod.build_embeddings(self.conn, only_missing=True, model_name="test-embed")
            second = aimod.build_embeddings(self.conn, only_missing=True, model_name="test-embed")
        self.assertEqual(first["processed"], 1)
        self.assertEqual(second["processed"], 0)

    def test_generate_distill_samples_reports_dedupe_and_teacher_success(self):
        doc_id = dbmod.get_doc_id_by_url(self.conn, "https://example.com/page")
        aimod.sync_document_pipeline(
            self.conn,
            doc_id=doc_id,
            url="https://example.com/page",
            title="Example",
            raw_html="<html>v1</html>",
            content="Bu bir ornek dokumandir. Icindeki bilgiler tekrar kullanilabilir. " * 8,
            language="tr",
            content_hash="hash-v1",
        )
        teacher_json = '{"question":"Ornek belge ne anlatiyor?","answer":"Ornek belge tekrar kullanilabilir bilgiler anlatiyor."}'
        with mock.patch("donersearch.ai_platform._openrouter_chat", return_value=(teacher_json, "", "teacher/mock")):
            first = aimod.generate_distill_samples(self.conn, dataset_version="distill_test", limit=1, teacher_model="teacher/mock")
            second = aimod.generate_distill_samples(self.conn, dataset_version="distill_test", limit=1, teacher_model="teacher/mock")
        self.assertEqual(first["created"], 1)
        self.assertEqual(first["teacher_success_count"], 1)
        self.assertEqual(first["fallback_count"], 0)
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
