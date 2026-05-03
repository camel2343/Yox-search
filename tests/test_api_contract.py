import io
import json
import os
import tempfile
import unittest
from unittest import mock
from wsgiref.util import setup_testing_defaults

from donersearch import ai_platform as aimod
from donersearch import db as dbmod
from donersearch.indexer import index_document
from donersearch.web import app_factory


def _request_json(app, path, params=None):
    environ = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    if params:
        from urllib.parse import urlencode

        environ["QUERY_STRING"] = urlencode(params)
    environ["wsgi.input"] = io.BytesIO(b"")
    captured = {"status": "500 Internal Server Error"}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status

    body = b"".join(app(environ, start_response))
    return int(captured["status"].split()[0]), json.loads(body.decode("utf-8"))


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        conn = dbmod.open_db(self.tmp.name)
        dbmod.ensure_schema(conn)
        self.doc_id, _changed = index_document(
            conn,
            "https://example.com/page",
            "Ornek Belge",
            "Bu belge apitestbenzersiz kelimesiyle doner search API testleri icin ornek bir icerik saglar. " * 10,
            language="tr",
            force=True,
        )
        aimod.publish_model(
            conn,
            model_name="student",
            model_version="v1",
            provider="local",
            artifact_path="data/models/student/v1",
            config={"smoke": True},
            activate=True,
        )
        conn.close()
        self.app = app_factory(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            try:
                os.unlink(self.tmp.name)
            except PermissionError:
                pass

    def test_api_endpoints_return_expected_shapes(self):
        search_result = ([(self.doc_id, 1.23)], {}, ["apitestbenzersiz"])
        with mock.patch("donersearch.web.search_with_fuzzy", return_value=search_result):
            with mock.patch("donersearch.ai_platform.search_with_fuzzy", return_value=search_result):
                with mock.patch("donersearch.ai_platform._openrouter_chat", return_value=("Kisa cevap", "", "teacher/mock")):
                    status_search, payload_search = _request_json(self.app, "/api/search", {"q": "apitestbenzersiz"})
                    status_sources, payload_sources = _request_json(self.app, "/api/sources", {"q": "apitestbenzersiz"})
                    status_answer, payload_answer = _request_json(self.app, "/api/answer", {"q": "apitestbenzersiz"})
            status_models, payload_models = _request_json(self.app, "/api/models")

        self.assertEqual(status_search, 200)
        self.assertEqual(payload_search["query"], "apitestbenzersiz")
        self.assertGreaterEqual(len(payload_search["results"]), 1)

        self.assertEqual(status_sources, 200)
        self.assertEqual(payload_sources["query"], "apitestbenzersiz")
        self.assertGreaterEqual(len(payload_sources["sources"]), 1)

        self.assertEqual(status_answer, 200)
        self.assertEqual(payload_answer["provider"], "teacher/mock")
        self.assertTrue(payload_answer["answer"])
        self.assertGreaterEqual(len(payload_answer["sources"]), 1)

        self.assertEqual(status_models, 200)
        self.assertGreaterEqual(len(payload_models["models"]), 1)
        self.assertEqual(payload_models["models"][0]["model_name"], "student")


if __name__ == "__main__":
    unittest.main()
