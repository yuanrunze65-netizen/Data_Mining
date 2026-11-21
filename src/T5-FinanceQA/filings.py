
"""Build and search a CIK-scoped index of the supplied SEC HTML filings."""
import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sqlite3
from time import perf_counter

INDEX_VERSION = 1
TOKEN = re.compile(r"[a-z0-9]+", re.I)
STOP = set("a an and are as at be by can did do for from had has have how in into is it its of on or our the their this to was were what when which who will with year company most recent".split())
RELATED = [
    ("revenue", "sales", "turnover"), ("profits", "profitability", "income", "loss", "earnings", "margin"),
    ("products", "technology", "platform", "innovation"), ("cash", "liquidity", "capital"),
    ("debt", "borrowings", "notes", "financing"), ("fx", "foreign", "currency", "exchange"),
    ("buyback", "repurchase", "dividend"), ("deferral", "deferred", "unearned"),
    ("commodity", "fuel", "diesel"), ("investment", "expenditure", "capex"),
]
HIDE = {"script", "style", "head", "title", "ix:header", "ix:hidden"}
SECTION = re.compile(r"^(?:part\s+[IVX]+\b|item\s+\d+[a-z]?(?:\s*[.:]|\s)|note\s+\d+(?:\s*[.:—–-]|\s))", re.I)

def words(s):
    return list(dict.fromkeys(w for w in TOKEN.findall(s.lower()) if w not in STOP and len(w) > 1))[:35]

def clean(s):
    return re.sub(r"\s+", " ", s.replace("\u200b", "")).strip()

class FilingParser(HTMLParser):
    """Preserve prose, heading paths, and row boundaries without loading full HTML into the model."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.parts, self.cell, self.row, self.table = [], [], [], [], []
        self.hidden, self.depth, self.in_cell, self.in_row = [], 0, False, False
        self.heading, self.anchor = "", ""

    def prose(self):
        text = clean("".join(self.parts))
        self.parts.clear()
        if not text:
            return
        heading = bool(len(text) < 125 and SECTION.search(text))
        if heading:
            self.heading = text
        self.blocks.append((self.heading, self.anchor, "text", text))

    def handle_starttag(self, tag, attrs):
        if self.hidden or tag in HIDE or any(
                (k == "style" and re.search(r"display\s*:\s*none", v or "", re.I))
                or k == "hidden" for k, v in attrs):
            if tag not in {"meta", "link", "br", "hr", "img", "input", "col", "wbr"}:
                self.hidden.append(tag)
            return
        if tag == "table":
            if not self.depth:
                self.prose()
                self.table = []
            self.depth += 1
        elif self.depth and tag == "tr":
            if not self.in_row:
                self.row = []
                self.in_row = True
        elif self.depth and tag in {"td", "th"}:
            if not self.in_cell:
                self.cell = []
                self.in_cell = True
        elif self.depth:
            if tag in {"div", "p", "br"} and self.in_cell:
                self.cell.append(" ")
        elif tag in {"br", "div", "p", "li", "h1", "h2", "h3", "h4", "section"}:
            self.prose()
            anchor = dict(attrs).get("id")
            if anchor:
                self.anchor = anchor

    def handle_endtag(self, tag):
        if self.hidden:
            if tag in self.hidden:
                index = len(self.hidden) - 1 - self.hidden[::-1].index(tag)
                del self.hidden[index:]
            return
        if self.depth:
            if tag in {"td", "th"} and self.in_cell:
                self.row.append(clean("".join(self.cell)))
                self.in_cell = False
            elif tag == "tr" and self.in_row:
                cells = [x for x in self.row if x]
                if cells:
                    self.table.append(" | ".join(cells))
                self.in_row = False
            elif tag == "table":
                self.depth -= 1
                if not self.depth and self.table:
                    self.blocks.append((self.heading, self.anchor, "table", "\n".join(self.table)))
        elif tag in {"div", "p", "li", "h1", "h2", "h3", "h4", "section"}:
            self.prose()

    def handle_data(self, data):
        if not self.hidden:
            if self.in_cell:
                self.cell.append(data)
            elif not self.depth:
                self.parts.append(data)

    def close(self):
        super().close()
        self.prose()

def chunks(blocks):
    group, size, context = [], 0, ("", "", "text")
    for section, anchor, kind, text in blocks:
        if kind == "table":
            if group:
                yield (*context, "\n".join(group))
                group, size = [], 0
            lines = text.splitlines()
            # Include the first rows as context for each segment of a long table.
            for offset in range(0, len(lines), 12):
                segment = lines[offset:offset + 12]
                if offset:
                    segment = lines[:min(3, offset)] + segment
                yield (section, anchor, kind, "\n".join(segment)[:4200])
            continue
        for offset in range(0, len(text), 1250):
            part = text[offset:offset + 1250]
            if group and (section != context[0] or size + len(part) > 1650):
                yield (*context, "\n".join(group))
                group, size = [], 0
            if not group:
                context = (section, anchor, kind)
            group.append(part)
            size += len(part) + 1
    if group:
        yield (*context, "\n".join(group))

def read_only(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)

def build(data, index):
    """Unsupervised preparation; only the supplied filings directory is traversed."""
    tick = perf_counter()
    paths = sorted((data / "filings").glob("*/*.htm*"))
    if not paths or any(not p.parent.name.isdigit() for p in paths):
        raise ValueError("No CIK-scoped report HTML files were found")
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_suffix(".building")
    temporary.unlink(missing_ok=True)
    manifest = []
    with sqlite3.connect(temporary) as db:
        db.executescript("""
        CREATE TABLE doc (id INTEGER PRIMARY KEY, cik TEXT, file TEXT, period TEXT,
                          pos INTEGER, section TEXT, anchor TEXT, kind TEXT, text TEXT);
        CREATE INDEX doc_scope ON doc(cik, file, pos);
        CREATE VIRTUAL TABLE search USING fts5(section, text, tokenize='porter unicode61');
        """)
        for path in paths:
            raw = path.read_bytes()
            parser = FilingParser()
            parser.feed(raw.decode("utf-8-sig", errors="replace"))
            parser.close()
            stamp = re.search(r"-(\d{4})(\d{2})(\d{2})", path.name)
            date = "-".join(stamp.groups()) if stamp else ""
            count = 0
            for count, (section, anchor, kind, text) in enumerate(chunks(parser.blocks), 1):
                cursor = db.execute("INSERT INTO doc (cik,file,period,pos,section,anchor,kind,text) VALUES (?,?,?,?,?,?,?,?)",
                                    (path.parent.name, path.name, date, count, section, anchor, kind, text))
                db.execute("INSERT INTO search(rowid,section,text) VALUES (?,?,?)", (cursor.lastrowid, section, text))
            if count < 50:
                raise ValueError(f"Too few segments parsed from {path.name}: {count}")
            manifest.append({"path": path.relative_to(data).as_posix(),
                             "sha256": hashlib.sha256(raw).hexdigest(), "chunks": count})
    # On Windows a SQLite connection must be explicitly closed before renaming.
    db.close()
    temporary.replace(index)
    return {"index_version": INDEX_VERSION, "elapsed_s": round(perf_counter() - tick, 3),
            "index_bytes": index.stat().st_size, "filings": manifest}

class Reports:
    def __init__(self, index, cik):
        self.index, self.cik = index, cik

    @staticmethod
    def evidence(row, max_chars=2400):
        identifier, file, date, pos, section, anchor, kind, text = row
        return dict(id=f"H{identifier}", file=file, period=date, chunk=pos,
                    section=section, html_anchor=anchor, kind=kind, text=text[:max_chars])

    def catalog(self):
        with read_only(self.index) as db:
            return [dict(file=file, period=date, chunks=count) for file, date, count in db.execute(
                "SELECT file,period,count(*) FROM doc WHERE cik=? GROUP BY file,period ORDER BY period DESC",
                (self.cik,))]

    def search(self, query, mode, year=None, kind=None, limit=5):
        if not isinstance(query, str) or len(query) > 500:
            raise ValueError("Search query must be a short string")
        tokens = words(query)
        if not tokens:
            raise ValueError("Search for named topics, metrics or disclosures")
        if mode == "hybrid":
            for family in RELATED:
                if any(t in family for t in tokens):
                    tokens.extend(w for w in family if w not in tokens)
        match = " OR ".join('"' + w + '"' for w in tokens[:44])
        sql = """SELECT d.id,d.file,d.period,d.pos,d.section,d.anchor,d.kind,d.text,
                        bm25(search,0.35,1.0) AS rank
                  FROM search JOIN doc d ON d.id=search.rowid
                  WHERE search MATCH ? AND d.cik=?"""
        params = [match, self.cik]
        if year is not None:
            sql += " AND substr(d.period,1,4)=?"
            params.append(str(year))
        if kind is not None:
            if kind not in {"text", "table"}:
                raise ValueError("kind must be text or table")
            sql += " AND d.kind=?"
            params.append(kind)
        with read_only(self.index) as db:
            rows = db.execute(sql + " ORDER BY rank,d.id LIMIT 100", params).fetchall()
        if mode == "hybrid":
            original = words(query)
            def score(row):
                document = set(TOKEN.findall(row[7].lower()))
                matched = sum(t in document for t in original)
                recent = int(row[2][:4] or 0)
                toc = 4 if row[7].count("Page") >= 4 or row[4].lower().startswith("table of contents") else 0
                phrase = 2.5 if " ".join(original) in row[7].lower() else 0
                return matched * 2.1 + phrase + min(18, -row[-1]) * .48 + (recent - 2020) * .24 - toc
            rows.sort(key=score, reverse=True)
        selected, seen, sections = [], set(), Counter()
        for row in rows:
            key = clean(row[7]).casefold()
            if key in seen:
                continue
            if mode == "hybrid" and sections[(row[1], row[4])] >= 2 and row[4]:
                continue
            seen.add(key)
            sections[(row[1], row[4])] += 1
            selected.append(self.evidence(row[:-1]))
            if len(selected) >= min(max(int(limit), 1), 8):
                break
        return selected

    def read(self, chunk_id, radius=1):
        if not isinstance(chunk_id, str) or not re.fullmatch(r"H\d{1,8}", chunk_id):
            raise ValueError("Use a chunk ID such as H3 returned by search")
        radius = min(max(int(radius), 0), 2)
        with read_only(self.index) as db:
            row = db.execute("SELECT file,pos FROM doc WHERE cik=? AND id=?",
                             (self.cik, int(chunk_id[1:]))).fetchone()
            if row is None:
                raise ValueError("Chunk does not exist for this CIK")
            file, pos = row
            rows = db.execute("""SELECT id,file,period,pos,section,anchor,kind,text FROM doc
                    WHERE cik=? AND file=? AND pos BETWEEN ? AND ? ORDER BY pos""",
                    (self.cik, file, pos - radius, pos + radius)).fetchall()
        return [self.evidence(row, 3200) for row in rows]
