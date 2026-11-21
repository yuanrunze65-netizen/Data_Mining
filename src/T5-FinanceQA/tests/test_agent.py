
"""Offline checks for report parsing, grounding, company scope, and finance tools."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filings import FilingParser, chunks, Reports
from run import answer_one, completion_url, parse, DEFAULT_INDEX, DEFAULT_PACK
from tools import Toolset, arithmetic

class ParserTest(unittest.TestCase):
    def test_hidden_data_and_table_boundaries(self):
        p = FilingParser()
        p.feed('''<head><style>hidden</style></head><div id="market">Item 7A. Market Risk</div>
          <p>Exposure to currency rates.</p><ix:hidden>DO NOT INDEX</ix:hidden>
          <table><tr><th>Operating revenue</th><th>2024</th></tr><tr>
          <td>Revenue</td><td>110</td></tr></table><p>After table.</p>''')
        p.close()
        c = list(chunks(p.blocks))
        self.assertTrue(any(k == "table" and "Operating revenue | 2024\nRevenue | 110" in s
                            and a == "market" for _, a, k, s in c))
        self.assertFalse(any("DO NOT INDEX" in s or "hidden" in s for _, _, _, s in c))

    def test_provider_json_repair_preserves_boundary(self):
        raw = '{"answer":"First sentence.' + chr(10) + 'Second sentence.","sources":["H1"]}'
        answer = parse(raw)
        self.assertEqual(answer["answer"].splitlines(), ["First sentence.", "Second sentence."])
        self.assertEqual(json.loads(json.dumps(answer)), answer)
        with self.assertRaises(ValueError):
            parse(raw + ' and trailing content')

    def test_calculator_and_endpoint(self):
        self.assertAlmostEqual(arithmetic("(new/old-1)*100", {"new":108,"old":90}), 20)
        for value in ["round(x)", "x[0]", "True", "9**999", "1e99"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                arithmetic(value, {"x":100})
        with self.assertRaises(ZeroDivisionError):
            arithmetic("10/x", {"x":0})
        self.assertEqual(completion_url("https://api.deepseek.com"), "https://api.deepseek.com/v1/chat/completions")
        with self.assertRaises(ValueError):
            completion_url("https://provider.example/?key=forbidden")

@unittest.skipUnless(DEFAULT_INDEX.exists() and (DEFAULT_PACK / "agent_workspace/data/finance.db").exists(),
                     "Run prepare with the supplied FinanceQA data first")
class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.data = DEFAULT_PACK / "agent_workspace/data"
        self.t = Toolset(self.data, DEFAULT_INDEX, "1640147", "hybrid")

    def test_financial_rows_scoped_and_read_only(self):
        rows = self.t.call("sql", {"query": "SELECT cik,name FROM companies"})["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cik"], "1640147")
        self.assertEqual(self.t.call("sql", {"query": "SELECT count(cik) AS n FROM companies"})["rows"][0]["n"], 1)
        with self.assertRaises(sqlite3.DatabaseError):
            self.t.call("sql", {"query": "SELECT cik FROM main.companies"})
        with self.assertRaises(ValueError):
            self.t.call("sql", {"query": "DELETE FROM financial_facts"})
        self.assertTrue(any("Revenue" in r["tag"] for r in self.t.call("metrics", {"query": "revenue"})))

    def test_annual_dedupe_and_filing_scope(self):
        annual = self.t.call("annual", {
            "tags": ["RevenueFromContractWithCustomerExcludingAssessedTax"], "years": [2023, 2024, 2025]})
        rows = annual["rows"]
        self.assertTrue(rows)
        keys = {(r["end_date"], r["unit"], r["fact_category"], r["fact_value"]) for r in rows}
        self.assertEqual(len(rows), len(keys))
        self.assertTrue(all(r["period_basis"] and r["accession_number"] and r["id"].startswith("F") for r in rows))
        hits = self.t.call("search", {"query": "revenue", "year": 2025, "limit": 3})
        self.assertTrue(hits)
        self.assertTrue(all(x["file"] == "snow-20250131.htm" for x in hits))
        with self.assertRaises(ValueError):
            Reports(DEFAULT_INDEX, "1373715").read(hits[0]["id"])

    def test_adaptive_grounding_loop(self):
        q = {"question_id": "TRAIN-PROTOCOL", "company_cik": "1640147", "question": "Find filing context."}
        configs = {"variant": "hybrid", "max_turns": 4, "max_tools_per_turn": 3, "tool_result_char_cap": 5200}
        step = 0
        def fake(cfg, key, messages):
            nonlocal step
            step += 1
            if step == 1:
                action = {"answer":"Wrong source.","sources":["H999999"]}
            elif step == 2:
                self.assertIn("Unknown source ID", messages[-1]["content"])
                action = {"calls":[{"tool":"search","args":{"query":"operating results","limit":1}}]}
            else:
                data = json.loads(messages[-1]["content"].split("\n", 1)[1])
                action = {"answer":"The filing discusses operating results.", "sources":[data[0]["result"][0]["id"]]}
            return json.dumps(action), {"prompt_tokens":5}, "stop", 1, 0.01
        prediction, trace = answer_one(q, configs, "", self.data, DEFAULT_INDEX, invoke=fake)
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(len(trace["calls"]), 1)
        self.assertEqual(len(trace["model_turns"]), 3)
        self.assertTrue(prediction["answer"])
        self.assertTrue(trace["sources"][0].startswith("H"))

if __name__ == "__main__":
    unittest.main()
