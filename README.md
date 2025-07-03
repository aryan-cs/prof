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

The tool operates in three main modes: `scrape`, `analyze`, and `outreach`.

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
- `/findcontact "<name_or_email>"`: Finds contact info (Website, LinkedIn, Google Scholar, Email) and papers for a specific author.
  - Example by name: `/findcontact "John Doe"`
  - Example by email: `/findcontact "j.doe@university.edu"`
- `/getcontacts <k> ["inst1"] ["inst2"]... [-save [filename.csv]] [--send-email]`: Scrapes contact info.
    - Gets the top `k` authors from each specified institution/group.
    - If no institution is given, it gets the top `k` authors overall.
    - `-save [filename.csv]`: Optionally saves the contacts to a CSV file (defaults to `contacts.csv`).
    - `--send-email`: After scraping, prompts to send outreach emails immediately.
  - **Examples:**
    - Get top 5 authors overall: `/getcontacts 5`
    - Get top 3 authors from Google and top 3 from Stanford University: `/getcontacts 3 "Google" "Stanford University"`
    - Get top 2 from Stanford, save to `stanford_contacts.csv`, and send emails: `/getcontacts 2 "Stanford University" -save stanford_contacts.csv --send-email`
- `/help`: Display the list of available commands.
- `/clear`: Clear the terminal screen.
- `/exit`: Exit the interactive analysis tool.

### 3. Outreach Mode
This mode sends outreach emails based on a contacts CSV file. Note that all PDFs stored in the `/mail` subdirectory will be sent as attachments to the email outlined by the template. Also note that the first line in the text file will be used as the subject, and all subsequent lines for the body.

**Command:**
```bash
python research.py outreach
```

**Arguments:**
- `--contacts-file`: (Optional) CSV file with contact information. Defaults to `contacts.csv`.
- `--email-template`: (Optional) Text file with the email template. Defaults to `mail/template.txt`.
- `--prof`: (Optional) Use 'Professor' as the title in the email salutation.
- `--test <email>`: (Optional) Send all emails to a test address instead of the actual recipients.

**Example:**
```bash
python research.py outreach --test mytestemail@example.com
```