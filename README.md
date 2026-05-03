# DonerSearch — Lightweight Web Crawler & Search Engine

A clean, dependency-free Python project that crawls web pages, indexes them into a SQLite database, and provides a search interface (CLI & Web) using BM25 ranking.

## Features

- **Crawler**: Follows links from seed URLs, respects basic robots.txt, and applies polite delays.
- **Indexing**: Uses SQLite with `documents`, `terms`, and `postings` tables.
- **Ranking**: BM25 scoring with snippets.
- **Interfaces**: Fully functional CLI and a built-in web server.
- **Zero External Dependencies**: Works out-of-the-box with Python 3.10+.
- **Image Metadata**: Optional image metadata extraction and deduplication.

## Getting Started

### Prerequisites

- Python 3.10 or higher.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/camel2343/Yox-search.git
   cd Yox-search
   ```
2. No additional installation required!

## Usage

The project includes a shortcut script `t` (on Windows) or you can use the module directly via `python -m donersearch`.

### 1. Advanced Crawling & Indexing

The `crawl` command is highly configurable. You can combine multiple parameters to control the behavior of the crawler.

#### Basic Crawl
```bash
python t crawl https://www.python.org --max-pages 50 --db doner.db
```

#### Professional Crawl (Multi-parameter)
Control depth, delay, and domain restrictions:
```bash
python t crawl https://news.ycombinator.com \
    --max-pages 100 \
    --max-depth 2 \
    --delay 1.0 \
    --allowed-domains ycombinator.com \
    --user-agent "MyCustomCrawler/1.0" \
    --db doner.db
```

#### Automated Discovery & Presets
Use presets for quick configuration or enable auto-discovery to find new domains:
```bash
# Using a preset (options: fast, default, deep)
python t crawl https://example.com --preset deep --db doner.db

# Enable auto-discovery of external links
python t crawl https://example.com --auto-discover --max-auto-discover-per-host 5 --db doner.db
```

#### Distributed/Background Crawling
Run with multiple workers and in a loop for continuous indexing:
```bash
python t crawl https://seed-url.com --loop --cycle-sleep 3600 --workers 4 --db doner.db
```

#### Crawler Command Options:
- `--max-pages`: Total limit of pages to index.
- `--max-depth`: How many links deep to follow from the seeds.
- `--delay`: Seconds to wait between requests (politeness).
- `--allowed-domains`: Restrict crawling to specific domains.
- `--auto-discover`: Automatically add new domains found in links to the queue.
- `--workers`: Number of parallel crawler threads.
- `--render`: Enable JavaScript rendering (requires Playwright).
- `--preset`: Use pre-defined settings (`fast`, `default`, `deep`).

### 2. Search via CLI
```bash
python t search "asyncio" --db doner.db
```

### 3. Start Web Interface
```bash
python t serve --host 127.0.0.1 --port 8000 --db doner.db
# Open http://127.0.0.1:8000 in your browser
```

## Directory Structure

- `donersearch/`: Core logic (crawler, indexer, search engine, web server).
- `data/`: Placeholder for crawled data, images, and models (ignored by git).
- `t`: Shortcut script for running commands.
- `README.md`: This documentation.

## License

MIT License. See `LICENSE` for details.
