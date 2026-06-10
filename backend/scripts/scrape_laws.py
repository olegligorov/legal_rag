"""
Scrapes Bulgarian laws from lex.bg and saves them as structured JSON files.

Each JSON file is a list of article objects:
  {
    "law_id": str,       # law title used as identifier
    "chapter": str,      # current chapter heading (empty string if none)
    "section": str,      # current section heading (empty string if none)
    "article": str,      # article number, e.g. "Чл. 1"
    "title": str,        # article title (empty string if none)
    "content": str,      # full article text, paragraphs joined with newlines
    "cross_references": list[str]  # referenced article/law names
  }
"""

import json
import re
import ssl
import time
import urllib.request
from pathlib import Path

# macOS framework Python ships without system CA certs; skip verification
# for this one-shot scraper against a known Bulgarian government site.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


LAWS = [
    {
        "url": "https://lex.bg/laws/ldoc/1594373121",
        "filename": "kodeks_na_truda.json",
    },
    {
        "url": "https://lex.bg/laws/ldoc/2135513678",
        "filename": "zakon_zashtita_potrebiteli.json",
    },
    {
        "url": "https://lex.bg/laws/ldoc/2121934337",
        "filename": "zakon_zadazhenia_dogovori.json",
    },
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "documentation"


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; legal-rag-scraper/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return resp.read().decode("windows-1251", errors="replace")


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def clean(text: str) -> str:
    text = strip_tags(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_DV_ANNOTATION = re.compile(r"\s*\([^)]*ДВ,\s*бр\.[^)]*\)")


def remove_dv_annotations(text: str) -> str:
    return _DV_ANNOTATION.sub("", text).strip()


def extract_cross_references(html: str) -> list[str]:
    refs = re.findall(
        r'class="(?:SameDocReference|LegalDocReference|NewDocReference)"[^>]*>([^<]+)',
        html,
    )
    return [r.strip() for r in refs if r.strip()]


def parse_law(html: str) -> tuple[str, list[dict]]:
    # Extract law title
    title_match = re.search(r'id="DocumentTitle"[^>]*>.*?<p[^>]*>(.*?)</p>', html, re.DOTALL)
    law_id = clean(title_match.group(1)) if title_match else "UNKNOWN"

    articles = []
    current_chapter = ""
    current_section = ""

    # Walk through all structural elements in document order.
    # Each element is one of: Heading, Section, Article, FinalEdicts,
    # TransitionalFinalEdicts, AdditionalEdicts — all share the same
    # <div ... class="TYPE" id="..."> pattern.
    element_pattern = re.compile(
        r'<div[^>]+class="(Heading|Section|Article|FinalEdicts|TransitionalFinalEdicts|AdditionalEdicts)"'
        r'[^>]*id="([^"]+)"[^>]*>(.*?)(?=<div[^>]+class="(?:Heading|Section|Article|FinalEdicts|'
        r'TransitionalFinalEdicts|AdditionalEdicts|buttons_)")',
        re.DOTALL,
    )

    for m in element_pattern.finditer(html):
        kind = m.group(1)
        content_html = m.group(3)

        if kind == "Heading":
            current_chapter = clean(content_html)
            current_section = ""
            continue

        if kind == "Section":
            current_section = clean(content_html)
            continue

        # Article / FinalEdicts / TransitionalFinalEdicts / AdditionalEdicts
        # Article title is in <p class="Title">
        article_title_match = re.search(r'<p class="Title">(.*?)</p>', content_html, re.DOTALL)
        article_title = clean(article_title_match.group(1)) if article_title_match else ""

        # Article number is in <b>Чл. N.</b> or similar bold tag
        article_num_match = re.search(r"<b>(Чл\.[^<]*|§\s*\d+[^<]*)</b>", content_html)
        article_num = clean(article_num_match.group(1)) if article_num_match else ""

        if not article_num:
            # For FinalEdicts sections without a specific article number, use the title
            article_num = article_title
            article_title = ""

        if not article_num:
            continue

        # Strip buttons block, article title, and any script tags
        body_html = re.sub(r"<p class=buttons>.*?</p>", "", content_html, flags=re.DOTALL)
        body_html = re.sub(r'<p class="Title">.*?</p>', "", body_html, flags=re.DOTALL)
        body_html = re.sub(r"<script[^>]*>.*?</script>", "", body_html, flags=re.DOTALL)

        # Collect cross-references before stripping tags
        cross_refs = extract_cross_references(body_html)

        # Extract individual paragraphs: each <div>...</div> block
        paragraphs = []
        for div_m in re.finditer(r"<div>(.*?)</div>", body_html, re.DOTALL):
            text = clean(div_m.group(1))
            if text:
                paragraphs.append(text)

        if not paragraphs:
            # Fallback: strip all tags from remaining body
            text = clean(body_html)
            if text:
                paragraphs = [text]

        content = remove_dv_annotations("\n".join(paragraphs))

        # Skip repealed/empty articles — nothing left after annotation stripping
        body = re.sub(r"^Чл\.\s*\S+\s*", "", content).strip()
        if not body:
            continue

        articles.append(
            {
                "law_id": law_id,
                "chapter": current_chapter,
                "section": current_section,
                "article": article_num,
                "title": article_title,
                "content": content,
                "cross_references": cross_refs,
            }
        )

    return law_id, articles


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for law in LAWS:
        print(f"Fetching {law['url']} ...")
        html = fetch(law["url"])

        law_id, articles = parse_law(html)
        print(f"  Parsed '{law_id}': {len(articles)} articles")

        out_path = OUTPUT_DIR / law["filename"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"  Saved -> {out_path}")
        time.sleep(1)  # be polite to the server

    print("Done.")


if __name__ == "__main__":
    main()
