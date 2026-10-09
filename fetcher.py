#!/usr/bin/env python3
"""
Pulsebot — LinkedIn Newsletter Fetcher (Piece 1)

Retrieves articles from a LinkedIn newsletter using multiple strategies:
  1. Direct listing-page crawl
  2. DuckDuckGo search-based discovery
  3. Manual URL list
  4. LinkedIn data-export ZIP (Articles & Activity)

Usage:
    python fetcher.py --newsletter-url "https://www.linkedin.com/newsletters/name-123/" \
                      --author "lenny-rachitsky" \
                      --out ./data

    python fetcher.py --export linkedin_export.zip --author "lenny-rachitsky" --out ./data
"""

import re
import sys
import time
import zipfile
import argparse
from pathlib import Path
from typing import Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Fetch one article ──────────────────────────────────────────────────────────

def fetch_article(url: str, retries: int = 2) -> Optional[dict]:
    """Fetch and extract text from a single LinkedIn article URL."""
    for attempt in range(retries + 1):
        try:
            with httpx.Client(headers=HEADERS, timeout=20, follow_redirects=True) as client:
                resp = client.get(url)
            if resp.status_code != 200:
                return None

            text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                no_fallback=False,
            )
            if not text or len(text.strip()) < 100:
                return None

            soup = BeautifulSoup(resp.text, "lxml")

            title_tag = soup.find("title")
            title = (title_tag.text.strip()
                     .replace(" | LinkedIn", "")
                     .replace(" - LinkedIn", "")
                     if title_tag else url)

            # Published date from meta tags
            date = ""
            for attr in ({"property": "article:published_time"},
                         {"name": "article:published_time"},
                         {"property": "og:article:published_time"}):
                meta = soup.find("meta", attr)
                if meta and meta.get("content"):
                    date = meta["content"][:10]
                    break

            return {"url": url, "title": title, "content": text, "date": date}

        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


# ── Discover article URLs ──────────────────────────────────────────────────────

def _urls_from_listing_page(newsletter_url: str, max_articles: int = 200) -> list[str]:
    """
    Pull article links from a LinkedIn newsletter listing page, following
    pagination via ?start=N until no new URLs appear or max_articles is reached.
    LinkedIn shows ~10 articles per page.
    """
    base_url = newsletter_url.rstrip("/").split("?")[0]
    seen: set[str] = set()
    urls: list[str] = []
    page_size = 10
    max_pages = (max_articles + page_size - 1) // page_size  # ceiling div

    try:
        with httpx.Client(headers=HEADERS, timeout=20, follow_redirects=True) as client:
            for page in range(max_pages):
                start = page * page_size
                page_url = base_url if page == 0 else f"{base_url}/?start={start}"

                resp = client.get(page_url)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, "lxml")
                new_on_page: list[str] = []

                for a in soup.find_all("a", href=True):
                    href: str = a["href"].split("?")[0].rstrip("/")
                    if not href:
                        continue
                    if href.startswith("/pulse/"):
                        href = "https://www.linkedin.com" + href
                    if "linkedin.com/pulse/" in href and href not in seen:
                        seen.add(href)
                        urls.append(href)
                        new_on_page.append(href)

                # No new articles on this page — pagination exhausted
                if not new_on_page:
                    break

                if page < max_pages - 1:
                    time.sleep(0.8)  # polite pause between pages

    except Exception:
        pass

    return urls


def _urls_from_search(author_identifier: str, max_results: int = 60) -> list[str]:
    """
    Use DuckDuckGo to find LinkedIn Pulse article URLs for an author.
    author_identifier: LinkedIn handle (e.g. 'lenny-rachitsky') or display name.
    """
    try:
        from ddgs import DDGS
        queries = [
            f'site:linkedin.com/pulse "{author_identifier}"',
            f'site:www.linkedin.com/pulse {author_identifier}',
        ]
        seen: set[str] = set()
        urls: list[str] = []
        per_query = max(10, max_results // 2)
        with DDGS() as d:
            for query in queries:
                for r in d.text(query, max_results=per_query):
                    href = r.get("href", "").split("?")[0].rstrip("/")
                    if "linkedin.com/pulse/" in href and href not in seen:
                        seen.add(href)
                        urls.append(href)
        return urls
    except Exception:
        return []


# ── LinkedIn data-export ingestion ────────────────────────────────────────────

def ingest_from_export(zip_path: str) -> list[dict]:
    """
    Parse articles from a LinkedIn data-export ZIP.
    Download from: LinkedIn → Settings → Data Privacy → Get a copy of your data
    → select 'Articles and Activity' → request archive.
    """
    articles: list[dict] = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            html_files = [f for f in z.namelist()
                          if re.search(r"Articles?/.*\.html?$", f, re.IGNORECASE)]
            if not html_files:
                console.print("[yellow]No article HTML files found in ZIP.[/yellow]")
                return []

            for filename in html_files:
                with z.open(filename) as f:
                    html = f.read().decode("utf-8", errors="replace")

                text = trafilatura.extract(html, include_comments=False, favor_precision=True)
                if not text or len(text.strip()) < 50:
                    continue

                soup = BeautifulSoup(html, "lxml")
                title_tag = soup.find("title")
                title = title_tag.text.strip() if title_tag else Path(filename).stem

                articles.append({
                    "url": "",
                    "title": title,
                    "content": text,
                    "date": "",
                    "source": "export",
                })

    except zipfile.BadZipFile:
        console.print(f"[red]Could not open ZIP: {zip_path}[/red]")
    except Exception as e:
        console.print(f"[red]Export error: {e}[/red]")

    return articles


# ── Main fetch orchestrator ────────────────────────────────────────────────────

def fetch_newsletter(
    newsletter_url:     Optional[str]       = None,
    author_identifier:  Optional[str]       = None,
    manual_urls:        Optional[list[str]] = None,
    export_zip:         Optional[str]       = None,
    max_articles:       int                 = 100,
    delay:              float               = 1.5,
) -> list[dict]:
    """
    Fetch all available articles for an author.
    Tries strategies in order until enough URLs are found.
    Returns list of {url, title, content, date} dicts.
    """
    candidate_urls: list[str] = []

    # 1 — manual list takes priority
    if manual_urls:
        candidate_urls = list(manual_urls)
        console.print(f"[dim]Using {len(candidate_urls)} manually provided URLs.[/dim]")

    # 2 — listing page
    if not candidate_urls and newsletter_url:
        console.print("[dim]Trying newsletter listing page…[/dim]")
        candidate_urls = _urls_from_listing_page(newsletter_url)
        console.print(f"[dim]  Found {len(candidate_urls)} URLs from listing page.[/dim]")

    # 3 — search-based discovery (supplement or fallback)
    if author_identifier and len(candidate_urls) < 5:
        console.print("[dim]Searching DuckDuckGo for articles…[/dim]")
        search_urls = _urls_from_search(author_identifier, max_results=max_articles)
        seen = set(candidate_urls)
        added = 0
        for u in search_urls:
            if u not in seen:
                candidate_urls.append(u)
                seen.add(u)
                added += 1
        console.print(f"[dim]  Found {added} additional URLs via search.[/dim]")

    # 4 — data export (run in parallel with URL fetching if provided)
    export_articles: list[dict] = []
    if export_zip:
        console.print(f"[dim]Reading LinkedIn data export: {export_zip}[/dim]")
        export_articles = ingest_from_export(export_zip)
        console.print(f"[dim]  Extracted {len(export_articles)} articles from export.[/dim]")

    if not candidate_urls and not export_articles:
        console.print("[red]No article sources found. Provide a newsletter URL, author handle, manual URLs, or export ZIP.[/red]")
        return []

    # Fetch each URL
    fetched: list[dict] = list(export_articles)
    total = min(len(candidate_urls), max_articles)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles", total=total)
        for i, url in enumerate(candidate_urls[:max_articles]):
            progress.update(task, description=f"[bold cyan]{url[-60:]}")
            article = fetch_article(url)
            if article:
                fetched.append(article)
            if i < total - 1:
                time.sleep(delay)
            progress.advance(task)

    console.print(f"\n[green]✓ Fetched {len(fetched)} articles.[/green]")
    return fetched


# ── Save to markdown ───────────────────────────────────────────────────────────

def save_as_markdown(articles: list[dict], output_dir: str, author_handle: str) -> list[str]:
    """Save articles as frontmatter-markdown files for downstream ingestion."""
    out = Path(output_dir) / author_handle
    out.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i, art in enumerate(articles):
        slug = re.sub(r"[^\w\-]", "_", art["title"][:60]).strip("_") or f"article_{i}"
        date_prefix = art.get("date", "undated")[:10]
        path = out / f"{date_prefix}_{slug}.md"

        with open(path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"title: \"{art['title'].replace(chr(34), chr(39))}\"\n")
            f.write(f"url: \"{art.get('url', '')}\"\n")
            f.write(f"date: \"{art.get('date', '')}\"\n")
            f.write(f"author: \"{author_handle}\"\n")
            f.write("---\n\n")
            f.write(art["content"])

        paths.append(str(path))

    console.print(f"[green]✓ Saved {len(paths)} markdown files → {out}[/green]")
    return paths


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a LinkedIn newsletter into markdown files.")
    parser.add_argument("--newsletter-url", help="LinkedIn newsletter page URL")
    parser.add_argument("--author",         help="LinkedIn handle or display name for search discovery")
    parser.add_argument("--urls",           nargs="+", help="Manual list of article URLs")
    parser.add_argument("--export",         help="Path to LinkedIn data-export ZIP")
    parser.add_argument("--out",            default="./data", help="Output directory (default: ./data)")
    parser.add_argument("--max",            type=int, default=100, help="Max articles to fetch (default: 100)")
    parser.add_argument("--delay",          type=float, default=1.5, help="Seconds between requests (default: 1.5)")
    args = parser.parse_args()

    if not any([args.newsletter_url, args.author, args.urls, args.export]):
        parser.error("Provide at least one of: --newsletter-url, --author, --urls, --export")

    author_handle = args.author or "unknown"

    console.print(f"\n[bold cyan]Pulsebot Fetcher[/bold cyan]  author=[cyan]{author_handle}[/cyan]\n")

    articles = fetch_newsletter(
        newsletter_url=args.newsletter_url,
        author_identifier=args.author,
        manual_urls=args.urls,
        export_zip=args.export,
        max_articles=args.max,
        delay=args.delay,
    )

    if not articles:
        console.print("[red]No articles retrieved.[/red]")
        sys.exit(1)

    paths = save_as_markdown(articles, args.out, author_handle)
    console.print(f"\n[bold green]Done.[/bold green] {len(paths)} files in {Path(args.out) / author_handle}\n")


if __name__ == "__main__":
    main()
