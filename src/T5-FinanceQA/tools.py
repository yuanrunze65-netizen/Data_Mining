
"""FinanceQA tools. Each Toolset is scoped to one company and one question."""
import ast
from contextlib import closing
from functools import lru_cache
import math
import operator
import re
import sqlite3

from filings import Reports, read_only, words

TABLES = ("companies", "company_tickers", "filings", "financial_facts")
FIELD_NOTE = """SQL tables (only the current CIK is visible):
companies(cik,name,fiscal_year_end,sic_description,...)
company_tickers(cik,ticker,exchange)
filings(cik,accession_number,form,filing_date,report_date,primary_document,...)
financial_facts(id,cik,fact_name,fact_value,unit,fact_category,fiscal_year,fiscal_period,
end_date,accession_number,form_type,filed_date,frame,...).
Select id and period/unit/provenance fields for fact evidence. fiscal_year labels the
source filing, not necessarily the observation. Dates and annual context matter."""
SCHEMA = """Tool arguments (all JSON objects):
catalog {} -> report inventory and validated annual ending dates
search {query, year?, kind?: "table"|"text", limit?} -> cite returned H IDs
read {chunk_id, radius?} -> retrieve adjacent report sections/table context
metrics {query, limit?} -> available XBRL concept tags, names and units
annual {tags: [exact tags], years?: [integer years], limit?} -> annual observations,
  point-in-time and duration qualifier, unresolved/conflicting records flagged
sql {query} -> read-only SELECT/WITH for this CIK; maximum 30 returned rows
calc {expression, variables?: {"name": number}} -> arithmetic +,-,*,/,** only;
  e.g. "(current/prior-1)*100", with variables from retrieved evidence.
  The result is C evidence; cite the input evidence as well."""
SQL_FUNCTIONS = set("""abs avg cast coalesce count date datetime glob group_concat ifnull
iif instr julianday length like lower ltrim max min nullif replace round rtrim
strftime substr substring sum total trim typeof upper printf concat row_number rank
dense_rank lag lead first_value last_value""".split())

ALIASES = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "operating income": ["OperatingIncomeLoss"], "cost": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "capital expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"], "dividend": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "liquidity": ["AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue"],
    "debt": ["LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt"],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "deferred revenue": ["ContractWithCustomerLiabilityCurrent", "ContractWithCustomerLiabilityNoncurrent"],
}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

def arithmetic(expression, variables):
    if not isinstance(expression, str) or len(expression) > 220 or not isinstance(variables, dict) or len(variables) > 16:
        raise ValueError("Use a short arithmetic expression and a small numeric variable map")
    def checked(x):
        if type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e40:
            raise ValueError("Invalid or out-of-bounds numeric input/result")
        return x
    def visit(node):
        if isinstance(node, ast.Constant):
            return checked(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return checked(variables[node.id])
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
            return checked(OPERATORS[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 12:
                raise ValueError("Exponent out of bounds")
            return checked(OPERATORS[type(node.op)](a, b))
        raise ValueError("Only arithmetic and named numeric variables are supported")
    return visit(ast.parse(expression, mode="eval").body)

def normalize(name):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).lower()

@lru_cache(maxsize=96)
def concepts(path, cik):
    with closing(read_only(path)) as db:
        return db.execute("""SELECT fact_name,unit,count(*) FROM financial_facts
                             WHERE cik=? GROUP BY fact_name,unit ORDER BY fact_name,unit""", (cik,)).fetchall()

@lru_cache(maxsize=96)
def annual_dates(path, cik):
    with closing(read_only(path)) as db:
        end = [(r[0], r[1]) for r in db.execute(
            """SELECT report_date,accession_number FROM filings
               WHERE cik=? AND form='10-K' AND report_date IS NOT NULL ORDER BY report_date DESC""", (cik,))
               if r[0] and re.fullmatch(r"\d{4}-\d\d-\d\d", r[0])]
        if not end:
            end = [(r[0], None) for r in db.execute("""SELECT DISTINCT end_date FROM financial_facts
                    WHERE cik=? AND form_type='10-K' AND frame GLOB 'CY[0-9][0-9][0-9][0-9]'
                    ORDER BY end_date DESC""", (cik,)) if r[0]]
    return dict(end)

class Toolset:
    def __init__(self, data, index, cik, variant):
        if not isinstance(cik, str) or not re.fullmatch(r"\d{1,10}", cik):
            raise ValueError("Invalid company CIK")
        self.path, self.cik = data / "finance.db", cik
        self.reports = Reports(index, cik)
        self.variant, self.evidence, self.calc_id = variant, {}, 0

    def call(self, name, params):
        if not isinstance(params, dict):
            raise ValueError("Tool args must be an object")
        methods = {"catalog": self.catalog, "search": self.search, "read": self.reports.read, "metrics": self.metrics,
                   "annual": self.annual, "sql": self.sql, "calc": self.calc}
        if name not in methods:
            raise ValueError("Unknown tool")
        result = methods[name](**params)
        rows = result.get("rows", []) if isinstance(result, dict) else result
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    self.evidence[row["id"]] = row
        return result

    def catalog(self):
        return {"reports": self.reports.catalog(),
                "annual_ends": sorted(annual_dates(self.path, self.cik), reverse=True)[:12]}

    def search(self, query, year=None, kind=None, limit=5):
        return self.reports.search(query, self.variant, year, kind, limit)

    def metrics(self, query, limit=12):
        if not isinstance(query, str) or len(query) > 240:
            raise ValueError("Use a short description of the metric")
        q = query.casefold()
        terms = words(q)
        tags = {}
        for tag, unit, n in concepts(self.path, self.cik):
            tag_words = normalize(tag)
            rank = sum(3 for t in terms if t in tag_words)
            if q == tag_words or q == tag.casefold():
                rank += 20
            if self.variant == "hybrid":
                for key, aliases in ALIASES.items():
                    if key in q and tag in aliases:
                        rank += 10 - aliases.index(tag)
            if rank:
                tags[tag] = (max(rank, tags.get(tag, (0,))[0]), (tags.get(tag, (0, [], 0))[1] + [unit]),
                             n + tags.get(tag, (0, [], 0))[2])
        ranked = sorted(tags.items(), key=lambda it: (-it[1][0], -it[1][2], it[0]))
        return [{"tag": k, "units": units, "records": n} for k, (_, units, n) in ranked[:max(1, min(int(limit), 18))]]

    def annual(self, tags, years=None, limit=25):
        if not isinstance(tags, list) or not 1 <= len(tags) <= 10 or any(
                not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,150}", t) for t in tags):
            raise ValueError("Provide 1–10 exact concept tags from metrics/schema")
        years = [int(y) for y in years] if isinstance(years, list) else None
        by_end = annual_dates(self.path, self.cik)
        year_set = set(years or sorted({int(d[:4]) for d in by_end}, reverse=True)[:4])
        slots = ",".join("?" for _ in tags)
        with closing(read_only(self.path)) as db:
            db.row_factory = sqlite3.Row
            raw = [dict(r) for r in db.execute(f"""SELECT id,fact_name,fact_value,unit,fact_category,
                  end_date,accession_number,form_type,filed_date,fiscal_year,fiscal_period,frame
                  FROM financial_facts WHERE cik=? AND fact_name IN ({slots}) AND
                  fact_value IS NOT NULL AND form_type IN ('10-K','10-K/A')""", (self.cik, *tags))]
        cohorts = {}
        for row in raw:
            date = row["end_date"]
            frame = row["frame"] or ""
            # January fiscal closes are reported by calendar closing year; the actual date is always returned.
            if not date or not re.fullmatch(r"\d{4}-\d\d-\d\d", date) or int(date[:4]) not in year_set:
                continue
            if re.fullmatch(r"CY\d{4}Q[1-4]", frame):
                continue
            if re.fullmatch(r"CY\d{4}", frame):
                qualifier = "documented annual duration"
            elif date not in by_end:
                continue
            elif frame.endswith("I"):
                qualifier = "point in time at annual end"
            else:
                qualifier = "annual end; duration unverified"
            row["period_basis"] = qualifier
            row["primary_filing"] = (by_end.get(date) is not None and row["accession_number"] == by_end[date])
            cohorts.setdefault((row["fact_name"], row["unit"], row["fact_category"], date), []).append(row)
        out = []
        for rows in cohorts.values():
            def preference(r):
                rank = {"documented annual duration": 2, "point in time at annual end": 2}.get(r["period_basis"], 0)
                if self.variant == "hybrid":
                    return rank, r["primary_filing"], r["filed_date"] or "", r["id"]
                return rank, r["filed_date"] or "", r["primary_filing"], r["id"]
            rows.sort(key=preference, reverse=True)
            seen = set()
            differing = []
            for item in rows:
                if item["fact_value"] in seen:
                    continue
                seen.add(item["fact_value"])
                differing.append({**item, "id": f'F{item["id"]}'})
            head = differing[0]
            head["distinct_values"] = len(differing)
            out.append(head)
            if len(differing) > 1:
                extra = differing[1]
                extra["alternative_disclosure"] = True
                out.append(extra)
        out.sort(key=lambda r: (tags.index(r["fact_name"]), -int(r["end_date"].replace("-", "")),
                                bool(r.get("alternative_disclosure")), r["unit"]))
        return {"rows": out[:max(1, min(int(limit), 30))],
                "note": "Years denote calendar closing dates. Conflicts may reflect periods or revisions; verify with filing tables."}

    def sql(self, query):
        if not isinstance(query, str) or len(query) > 4000 or not re.match(r"^\s*(select|with)\b", query, re.I):
            raise ValueError("Only a single SELECT/WITH statement is supported")
        cik = self.cik
        with closing(read_only(self.path)) as db:
            db.row_factory = sqlite3.Row
            for table in TABLES:
                db.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM main.{table} WHERE cik='{cik}'")
            def authorize(action, p1, p2, schema, via):
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and ((schema == "main" and p1 in TABLES and via == p1)
                        or (schema == "temp" and p1 in TABLES)):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and (p2 or "").lower() in SQL_FUNCTIONS:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            db.set_authorizer(authorize)
            remaining = iter(range(1000))
            db.set_progress_handler(lambda: int(next(remaining, None) is None), 2000)
            rows = db.execute("SELECT * FROM (" + query.rstrip().rstrip(";") + ") LIMIT 30").fetchall()
            result = []
            for row in rows:
                item = {k: (row[k][:360] if isinstance(row[k], str) else row[k]) for k in row.keys()}
                identifier = item.get("id")
                # Never treat an arbitrary SQL alias or a filings row as a fact ID.
                if type(identifier) is int and "fact_name" in item and "fact_value" in item:
                    fact = db.execute("SELECT id,fact_name,fact_value FROM financial_facts WHERE id=?",
                                      (identifier,)).fetchone()
                    if fact and fact["fact_name"] == item["fact_name"] and fact["fact_value"] == item["fact_value"]:
                        item["id"] = f"F{identifier}"
                result.append(item)
            return {"rows": result, "row_limit": 30}

    def calc(self, expression, variables=None):
        self.calc_id += 1
        return [{"id": f"C{self.calc_id}", "expression": expression, "variables": variables or {},
                 "value": arithmetic(expression, variables or {})}]
