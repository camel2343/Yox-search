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
   git clone https://github.com/yourusername/donersearch.git
   cd donersearch
   ```
2. No additional installation required!

### Usage

The project includes a shortcut script `t` (on Windows) or you can use the module directly.

#### 1. Crawl and Index
```bash
python t crawl https://www.python.org --max-pages 50 --max-depth 1 --delay 0.5 --db doner.db
```

#### 2. Search via CLI
```bash
python t search "asyncio" --db doner.db
```

#### 3. Start Web Interface
```bash
python t serve --host 127.0.0.1 --port 8000 --db doner.db
# Open http://127.0.0.1:8000 in your browser
```

## Directory Structure

- `donersearch/`: Core logic (crawler, indexer, search engine, web server).
- `data/`: (Ignored) Placeholder for crawled data, images, and models.
- `t`: Shortcut script for running commands.
- `README.md`: This documentation.

## License

MIT License. See `LICENSE` for details.
