import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from donersearch.__main__ import build_parser, main


class SmokeCliTests(unittest.TestCase):
    def test_smoke_e2e_requires_seed_and_query(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["smoke-e2e"])

    def test_smoke_e2e_invokes_orchestrator_and_prints_json(self):
        output = io.StringIO()
        expected = {"status": "ok", "seed_url": "https://example.com", "query": "python"}
        with mock.patch("donersearch.__main__.smokemod.run_smoke_e2e", return_value=expected) as mocked:
            with redirect_stdout(output):
                exit_code = main(["smoke-e2e", "--seed-url", "https://example.com", "--query", "python"])
        self.assertEqual(exit_code, 0)
        mocked.assert_called_once()
        self.assertEqual(json.loads(output.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
