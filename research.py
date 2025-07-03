#!/usr/bin/env python

import argparse
import asyncio
import functools
import os
import re
from dataclasses import dataclass
from googlesearch import search

import aiohttp
import bs4
import pandas as pd
from tqdm import tqdm

# --- Global variables for scraping ---
REQUESTS_PBAR: tqdm = None
OPEN_REQUESTS: asyncio.Semaphore = None
SPEAKER_ID_REGEX = re.compile(r"showSpeaker\('([\d-]+)'\)")

# --- Analysis Constants ---
LEADERBOARD_LENGTH = 10

# --- Scraping Functions ---

def retry_on_server_disconnect(n_tries: int):
    def decorator(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            for i in range(n_tries):
                try:
                    return await f(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if i == n_tries - 1:
                        print(f"A client error occurred: {e}")
                        raise
                    # await asyncio.sleep(2**i)  # Exponential backoff
        return wrapper
    return decorator

@retry_on_server_disconnect(5)
async def load_doc_from_url(session: aiohttp.ClientSession, url: str):
    global REQUESTS_PBAR, OPEN_REQUESTS
    if REQUESTS_PBAR is not None:
        REQUESTS_PBAR.total += 1
    async with OPEN_REQUESTS:
        async with session.get(url) as response:
            doc = bs4.BeautifulSoup(await response.text(), features="lxml")
            if REQUESTS_PBAR is not None:
                REQUESTS_PBAR.update()
            return doc

async def load_paper_ids(session: aiohttp.ClientSession, url):
    doc = await load_doc_from_url(session, url)
    cards = doc.select(".maincard.poster")
    return [c.attrs["id"][9:] for c in cards]

async def load_paper(session: aiohttp.ClientSession, url):
    doc = await load_doc_from_url(session, url)
    box = doc.select(".maincard")[0].parent
    title = box.select(".maincardBody")[0].text.strip()
    authors = [
        (b.text.strip(), SPEAKER_ID_REGEX.match(b.attrs["onclick"]).group(1))
        for b in box.find_all("button")
        if "showSpeaker" in b.attrs.get("onclick", "")
    ]
    return title, authors

async def load_author(session: aiohttp.ClientSession, url):
    doc = await load_doc_from_url(session, url)
    box = doc.select(".maincard")[0].parent
    name = box.find("h3").text.strip()
    affiliation = box.find("h4").text.strip()
    return name, affiliation

@dataclass
class Conference:
    name: str
    host: str
    first_year: int

    def papers_url(self, year: int):
        return f"https://{self.host}/Conferences/{year:d}/Schedule"

    def paper_url(self, year: int, id: str):
        return f"https://{self.host}/Conferences/{year:d}/Schedule?showEvent={id}"

    def author_url(self, year: int, id: str):
        return f"https://{self.host}/Conferences/{year:d}/Schedule?showSpeaker={id}"

    async def scrape(self, year: int, session: aiohttp.ClientSession):
        paper_ids = await load_paper_ids(session, self.papers_url(year))
        paper_links = [self.paper_url(year, id) for id in paper_ids]
        paper_tasks = [load_paper(session, link) for link in paper_links]
        paper_data = await asyncio.gather(*paper_tasks)

        author_ids = list(
            set([id for _, authors in paper_data for name, id in authors])
        )
        author_links = [self.author_url(year, id) for id in author_ids]
        author_tasks = [load_author(session, link) for link in author_links]
        author_data = await asyncio.gather(*author_tasks)
        affiliations = dict(author_data)

        papers = [
            (title, [(name, affiliations.get(name, "N/A")) for name, _ in authors])
            for title, authors in paper_data
        ]

        unnormalized = [
            (title, author, affiliation)
            for title, authors in papers
            for author, affiliation in authors
        ]

        papers_df = pd.DataFrame(unnormalized, columns=["Title", "Author", "Affiliation"])
        papers_df.insert(0, "Conference", self.name)
        papers_df.insert(1, "Year", year)
        return papers_df

CONFERENCES = [
    Conference("ICML", "icml.cc", 2017),
    Conference("NeurIPS", "neurips.cc", 2006),
    Conference("ICLR", "iclr.cc", 2018),
]

async def scrape_mode(args):
    global REQUESTS_PBAR, OPEN_REQUESTS

    output = args.output
    parallel = args.parallel
    years = args.years

    OPEN_REQUESTS = asyncio.Semaphore(parallel)

    if "-" in years:
        match = re.match(r"^(\d+)-(\d+)", years)
        if not match:
            print(f"Error: Invalid year range {years}; expected e.g. 2008-2010")
            return
        start, end = int(match[1]), int(match[2])
    else:
        start = end = int(years)
    year_range = range(start, end + 1)

    try:
        existing_df = pd.read_csv(output)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=["Conference", "Year", "Title", "Author", "Affiliation"])

    cf_names = ", ".join(c.name for c in CONFERENCES)
    print(f"Scraping papers from {start}-{end} in {cf_names} into {output}")

    with tqdm(total=0, desc="Overall Progress") as pbar:
        REQUESTS_PBAR = pbar
        timeout = aiohttp.ClientTimeout(total=60 * 5)  # 5 minute timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            paper_tasks = [
                conf.scrape(year, session)
                for conf in CONFERENCES
                for year in year_range
                if year >= conf.first_year
            ]
            results = await asyncio.gather(*paper_tasks)

    if results:
        new_df = pd.concat(results, ignore_index=True)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Deduplicate
        combined_df.drop_duplicates(
            subset=["Conference", "Year", "Title", "Author"],
            keep="first",
            inplace=True
        )
        
        combined_df.to_csv(output, index=False)
        print(f"\nSuccessfully saved data to {output}")
        print(f"Total entries: {len(combined_df)}")

# --- Analysis Function ---

def show_leaderboards(df, length):
    print("\n" + "="*55 + "\n")
    
    print(f"--- Top {length} Publishing Groups ---")
    top_affiliations = df.dropna(subset=['Affiliation'])['Affiliation'].value_counts().head(length)
    for i, (item, count) in enumerate(top_affiliations.items(), 1):
        print(f"{i}. {item}: {count}")
    
    print("\n" + "="*55 + "\n")
    
    print(f"--- Top {length} Institutions ---")
    school_keywords = ['university', 'college', 'school', 'institute', 'polytechnic', 'eth', 'epfl']
    school_regex = '|'.join(school_keywords)
    schools_df = df[df['Affiliation'].str.contains(school_regex, case=False, na=False)]
    top_schools = schools_df['Affiliation'].value_counts().head(length)
    for i, (item, count) in enumerate(top_schools.items(), 1):
        print(f"{i}. {item}: {count}")
        
    print("\n" + "="*55 + "\n")

    print(f"--- Top {length} Most Frequent Authors ---")
    top_authors = df['Author'].value_counts().head(length)
    for i, (item, count) in enumerate(top_authors.items(), 1):
        print(f"{i}. {item}: {count}")
    
    print("\n" + "="*55 + "\n")

def show_authors_from(df, institution, length):
    print(f"\n--- Top {length} Authors from {institution} ---")
    inst_df = df[df['Affiliation'].str.contains(institution, case=False, na=False)]
    if inst_df.empty:
        print(f"No authors found for institution matching '{institution}'.")
        return
    
    top_authors = inst_df['Author'].value_counts().head(length)
    for i, (item, count) in enumerate(top_authors.items(), 1):
        print(f"{i}. {item}: {count}")
    print("\n" + "="*55 + "\n")


async def get_contacts(authors_with_affiliations):
    print("\n#" + "-" * 50 + "#")
    async with aiohttp.ClientSession() as session:
        for i, (author, affiliation) in enumerate(authors_with_affiliations):
            if i > 0:
                delay = 5  # 5-second delay
                # print(f"\nWaiting for {delay} seconds to avoid rate-limiting...")
                # await asyncio.sleep(delay)

            print(f"\n[{i}] {author}")
            query = f'{author} {affiliation or ""} contact information'
            try:
                loop = asyncio.get_running_loop()
                search_results = await loop.run_in_executor(
                    None, functools.partial(search, query, num_results=5)
                )
                search_results = list(search_results)

                personal_site = search_results[0] if search_results else None
                print(f"  Website: {personal_site if personal_site else 'N/A'}")
                
                linkedin_results = [r for r in search_results if "linkedin.com/in" in r]
                print(f"  LinkedIn: {linkedin_results[0] if linkedin_results else 'N/A'}")

                email = 'N/A'
                if personal_site:
                    try:
                        # Use the session to get the personal site
                        async with session.get(personal_site, timeout=10) as response:
                            if response.status == 200:
                                text = await response.text()
                                
                                # First, try standard email regex
                                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                                found_emails = re.findall(email_pattern, text)
                                if found_emails:
                                    email = found_emails[0]
                                else:
                                    # If not found, try to de-obfuscate and then find
                                    deobfuscated_text = text.replace(' [at] ', '@').replace(' [dot] ', '.')
                                    # A simpler pattern for de-obfuscated text might be needed
                                    found_emails = re.findall(email_pattern, deobfuscated_text)
                                    if found_emails:
                                        email = found_emails[0]

                            else:
                                print(f"    Could not fetch {personal_site}, status: {response.status}")
                    except Exception as e:
                        print(f"    Could not scrape email from {personal_site}: {e}")

                print(f"  Email: {email.lower()}")

            except Exception as e:
                print(f"Could not fetch contact info for {author}: {e}")

            print("\n#" + "-" * 50 + "#")
        print()

async def analyze_mode(args):
    file_path = args.output
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_path}'. Found {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please run the 'scrape' mode first to generate the data.")
        return

    leaderboard_length = 10
    print("Welcome to the interactive analysis tool!")
    print("Type `/help` for a list of commands.")

    while True:
        try:
            command = input(">> ").strip()
            if not command:
                continue

            if command.startswith('/'):
                parts = command.split(maxsplit=1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == '/top':
                    try:
                        leaderboard_length = int(arg)
                        print(f"Leaderboard length set to {leaderboard_length}.")
                        show_leaderboards(df, leaderboard_length)
                    except ValueError:
                        print("Invalid number for /top command. Please use an integer.")
                elif cmd == '/from':
                    if not arg:
                        print("Please specify an institution for the /from command.")
                        continue
                    # Remove quotes if present
                    institution = arg.strip('"\'')
                    show_authors_from(df, institution, leaderboard_length)
                elif cmd == '/getcontacts':
                    parts = arg.split(maxsplit=1)
                    try:
                        k = int(parts[0])
                        institution = parts[1].strip('"\'') if len(parts) > 1 else None
                    except (ValueError, IndexError):
                        print("Invalid format. Use: /getcontacts <k> [\"institution\"]")
                        continue

                    if institution:
                        authors_df = df[df['Affiliation'].str.contains(institution, case=False, na=False)]
                    else:
                        authors_df = df
                    
                    top_authors_series = authors_df['Author'].value_counts().head(k)
                    authors_info = []
                    for author_name in top_authors_series.index:
                        # Find the most common affiliation for this author in the filtered df
                        author_affiliations = authors_df[authors_df['Author'] == author_name]['Affiliation']
                        most_common_affiliation = author_affiliations.dropna().mode()
                        if not most_common_affiliation.empty:
                            affiliation = most_common_affiliation[0]
                        else:
                            affiliation = None
                        authors_info.append((author_name, affiliation))

                    await get_contacts(authors_info)

                elif cmd == '/show':
                    show_leaderboards(df, leaderboard_length)
                elif cmd == '/help':
                    print("\nAvailable commands:")
                    print("  /show                  - Display all top leaderboards.")
                    print("  /top <number>          - Set the number of items in leaderboards.")
                    print("  /from \"<institution>\"  - Show top authors from an institution.")
                    print("  /getcontacts <k> [\"institution\"] - Scrape contact info for top k authors.")
                    print("  /clear                 - Clear the terminal screen.")
                    print("  /exit                  - Exit the interactive analysis tool.")
                elif cmd == '/exit':
                    break
                elif cmd == '/clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                else:
                    print(f"Unknown command: {cmd}")
            else:
                print("Commands must start with '/'. Type /help for a list of commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


# --- Main Execution ---

async def main():
    parser = argparse.ArgumentParser(
        description="Scrape and analyze paper data from ICML, NeurIPS, and ICLR."
    )
    parser.add_argument(
        "mode",
        choices=["scrape", "analyze"],
        help="The mode to run the script in: 'scrape' to gather data, 'analyze' to view statistics."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="papers.csv",
        help="File to store data. Used as input for analysis. [Default: papers.csv]",
    )
    # Arguments for scrape mode
    parser.add_argument(
        "--years",
        help="Year or year range (e.g., 2020 or 2018-2020). Required for 'scrape' mode.",
    )
    parser.add_argument(
        "--parallel",
        default=500,
        type=int,
        help="Number of parallel requests for scraping. [Default: 500]",
    )
    
    args = parser.parse_args()

    if args.mode == 'scrape':
        if not args.years:
            parser.error("argument --years is required for mode 'scrape'")
        await scrape_mode(args)
    elif args.mode == 'analyze':
        await analyze_mode(args)

if __name__ == "__main__":
    asyncio.run(main())
