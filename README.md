# prof

This tool is a Python-based command-line application for scraping and analyzing research papers from NeurIPS, ICML, and ICLR. It allows users to gather paper metadata (titles, authors, affiliations) and perform interactive analysis on the collected data. The purpose of this tool is to streamline the process of acquiring research opportunities for students interested in machine learning.  

The repository comes with papers from 2022-2024 pre-loaded, but earlier years can be scraped.

## Setup and Installation

1.  Clone the repository (or download the files)

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The tool operates in two main modes: `scrape` and `analyze`.

### 1. Scrape Mode

This mode is for gathering data from the conference websites.

**Command:**
```bash
python research.py scrape --years <YEAR_OR_RANGE>
```

**Arguments:**
- `--years`: (Required) The year or range of years to scrape.
  - For a single year: `--years 2022`
  - For a range of years: `--years 2020-2022`
- `--output`: (Optional) The name of the output CSV file. Defaults to `papers.csv`.
- `--parallel`: (Optional) The number of parallel requests to make. Defaults to `500`.

**Example:**
```bash
python research.py scrape --years 2021-2023
```

### 2. Analyze Mode

This mode provides an interactive shell for analyzing the data in the CSV file.

**Command:**
```bash
python research.py analyze
```

**Arguments:**
- `--output`: (Optional) The name of the CSV file to analyze. Defaults to `papers.csv`.

#### Interactive Commands

Once in analyze mode, you can use the following commands:

- `/show`: Display the top leaderboards for institutions, authors, and publishing groups.
- `/top <number>`: Set the number of entries to show in the leaderboards.
  - Example: `/top 15`
- `/from "<institution>"`: Show the top authors from a specific institution. The institution name should be in quotes.
  - Example: `/from "Google"`
- `/getcontacts <k> ["institution"]`: Scrape contact information for the top `k` authors. You can optionally filter by institution.
  - Example (top 5 overall authors): `/getcontacts 5`
  - Example (top 3 authors from "Stanford University"): `/getcontacts 3 "Stanford University"`
- `/help`: Display the list of available commands.
- `/clear`: Clear the terminal screen.
- `/exit`: Exit the interactive analysis tool.