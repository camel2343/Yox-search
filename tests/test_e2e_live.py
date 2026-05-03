import os
import tempfile
import unittest

from donersearch import smoke as smokemod


@unittest.skipUnless(os.environ.get("DONERSEARCH_LIVE_E2E") == "1", "live smoke disabled")
class LiveSmokeTests(unittest.TestCase):
    def test_live_smoke_pipeline(self):
        if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
            self.skipTest("OPENROUTER_API_KEY not configured")
        seed_url = os.environ.get("DONERSEARCH_LIVE_SEED_URL", "https://www.python.org")
        query = os.environ.get("DONERSEARCH_LIVE_QUERY", "Python")
        teacher_model = os.environ.get("DONERSEARCH_LIVE_TEACHER_MODEL")
        max_pages = int(os.environ.get("DONERSEARCH_LIVE_MAX_PAGES", "3"))
        max_depth = int(os.environ.get("DONERSEARCH_LIVE_MAX_DEPTH", "0"))
        distill_limit = int(os.environ.get("DONERSEARCH_LIVE_DISTILL_LIMIT", "1"))

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            result = smokemod.run_smoke_e2e(
                seed_url=seed_url,
                query=query,
                db_path=tmp.name,
                teacher_model=teacher_model,
                max_pages=max_pages,
                max_depth=max_depth,
                distill_limit=distill_limit,
            )
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["steps"]["crawl"]["pages_indexed"], 1)
            self.assertNotIn(
                result["steps"]["api_checks"]["answer_provider"],
                ("", "extractive_fallback", "none"),
            )
            self.assertGreaterEqual(result["steps"]["distill"]["first_run"]["teacher_success_count"], 1)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
