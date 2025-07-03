import argparse
import asyncio
import functools
import os
import re
from dataclasses import dataclass
from googlesearch import search
import smtplib
import ssl
import getpass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import aiohttp
import bs4
import pandas as pd
from tqdm import tqdm

REQUESTS_PBAR: tqdm = None
OPEN_REQUESTS: asyncio.Semaphore = None
SPEAKER_ID_REGEX = re.compile(r"showSpeaker\('([\d-]+)'\)")

LEADERBOARD_LENGTH = 10

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
            (title, [(name, affiliations.get(name, "n/a")) for name, _ in authors])
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
        timeout = aiohttp.ClientTimeout(total=60 * 5)
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
        
        combined_df.drop_duplicates(
            subset=["Conference", "Year", "Title", "Author"],
            keep="first",
            inplace=True
        )
        
        combined_df.to_csv(output, index=False)
        print(f"\nSuccessfully saved data to {output}")
        print(f"Total entries: {len(combined_df)}")

def show_leaderboards(df, length, which='all'):
    school_keywords = ['university', 'college', 'school', 'institute', 'polytechnic', 'eth', 'epfl', 'uc berkeley', 'mit', 'kaist', 'uiuc', 'ucla', 'cmu', 'politecnico di milano', 'uc san diego', 'universität']
    if which in ['all', 'groups']:
        print("\n" + "="*55 + "\n")
        print(f"--- Top {length} Publishing Groups ---")
        top_affiliations = df.dropna(subset=['Affiliation'])['Affiliation'].value_counts().head(length)
        for i, (item, count) in enumerate(top_affiliations.items(), 1):
            print(f"{i}. {item}: {count}")
    
    if which in ['all', 'schools']:
        print("\n" + "="*55 + "\n")
        print(f"--- Top {length} Institutions ---")
        school_regex = '|'.join(school_keywords)
        schools_df = df[df['Affiliation'].str.contains(school_regex, case=False, na=False)]
        top_schools = schools_df['Affiliation'].value_counts().head(length)
        for i, (item, count) in enumerate(top_schools.items(), 1):
            print(f"{i}. {item}: {count}")

    if which in ['all', 'companies']:
        print("\n" + "="*55 + "\n")
        print(f"--- Top {length} Companies ---")
        affiliations = df.dropna(subset=['Affiliation'])
        school_regex = '|'.join(school_keywords)
        companies_df = affiliations[~affiliations['Affiliation'].str.contains(school_regex, case=False, na=False)]
        top_companies = companies_df['Affiliation'].value_counts().head(length)
        for i, (item, count) in enumerate(top_companies.items(), 1):
            print(f"{i}. {item}: {count}")
        
    if which in ['all', 'authors']:
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
    # print("#" + "-" * 50 + "#")
    contacts_data = []
    async with aiohttp.ClientSession() as session:
        for i, (author, affiliation) in enumerate(authors_with_affiliations):
            if i > 0:
                await asyncio.sleep(2)

            print(f"\n[{i + 1}] {author}")
            query = f'{author} {affiliation or ""} contact email'
            try:
                loop = asyncio.get_running_loop()
                search_results = await loop.run_in_executor(
                    None, functools.partial(search, query, num_results=5, sleep_interval=2)
                )
                search_results = list(search_results)

                personal_site = 'n/a'
                linkedin_url = 'n/a'
                scholar_url = 'n/a'
                email = 'n/a'

                scholar_query = f'{author} {affiliation or ""} google scholar'
                try:
                    scholar_search_results = await loop.run_in_executor(
                        None, functools.partial(search, scholar_query, num_results=2, sleep_interval=1)
                    )
                    for r in scholar_search_results:
                        if "scholar.google.com/citations?user=" in r:
                            scholar_url = r
                            break
                except Exception:
                    pass


                linkedin_results = [r for r in search_results if "linkedin.com/in" in r]
                if linkedin_results:
                    linkedin_url = linkedin_results[0]

                for url in search_results:
                    if email != 'n/a':
                        break
                    
                    if personal_site == 'n/a' and not any(domain in url for domain in ['linkedin.com', 'scholar.google.com', 'dblp.org', 'twitter.com']):
                        personal_site = url

                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                text = await response.text()
                                soup = bs4.BeautifulSoup(text, 'html.parser')

                                mailto_links = soup.select('a[href^="mailto:"]')
                                if mailto_links:
                                    email = mailto_links[0]['href'][7:].split('?')[0]
                                    break 

                                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                                
                                deobfuscated_html = text.lower()
                                deobfuscated_html = deobfuscated_html.replace(' [at] ', '@').replace(' [dot] ', '.')
                                deobfuscated_html = deobfuscated_html.replace('(at)', '@').replace('(dot)', '.')
                                deobfuscated_html = deobfuscated_html.replace(' at ', '@').replace(' dot ', '.')
                                deobfuscated_html = deobfuscated_html.replace('&#64;', '@').replace('&#46;', '.')
                                deobfuscated_html = re.sub(r'\s*<span class="email">([^<]+)<\/span>\s*', r'\1', deobfuscated_html)


                                found_emails = re.findall(email_pattern, deobfuscated_html)
                                
                                if found_emails:
                                    last_name = author.split(' ')[-1].lower()
                                    preferred_emails = [e for e in found_emails if last_name in e]
                                    if preferred_emails:
                                        email = preferred_emails[0]
                                    else:
                                        email = found_emails[0]
                                    
                    except Exception as e:
                        pass
                
                print(f"  Website: {personal_site}")
                print(f"  LinkedIn: {linkedin_url}")
                print(f"  Google Scholar: {scholar_url}")
                print(f"  Email: {email.lower()}")

                contacts_data.append({
                    "Author": author,
                    "Affiliation": affiliation,
                    "Website": personal_site,
                    "LinkedIn": linkedin_url,
                    "Google Scholar": scholar_url,
                    "Email": email.lower()
                })

            except Exception as e:
                print(f"Could not fetch contact info for {author}: {e}")
                contacts_data.append({
                    "Author": author,
                    "Affiliation": affiliation,
                    "Website": "Error",
                    "LinkedIn": "Error",
                    "Google Scholar": "Error",
                    "Email": "Error"
                })

            # print("#" + "-" * 50 + "#")
        return contacts_data

async def send_outreach_email(server, sender_email, contact_info, papers_df, subject_template, email_body_template, prof_flag, test_email=None):
    author_name = contact_info["Author"]
    if author_name.isupper():
        author_name = author_name.title()

    last_name = author_name.split(' ')[-1]
    title = "Professor" if prof_flag else "Mr."

    author_papers = papers_df[papers_df['Author'] == author_name]
    if not author_papers.empty:
        most_recent_paper = author_papers.sort_values('Year', ascending=False).iloc[0]
        paper_title = most_recent_paper['Title']
    else:
        paper_title = "which you submitted to NeurIPS"

    subject = subject_template.replace("[TITLE]", title).replace("[LAST_NAME]", last_name)
    email_body = email_body_template.replace("[TITLE]", title).replace("[LAST_NAME]", last_name).replace("[MOST_RECENT_PAPER]", paper_title)
    
    recipient_email = contact_info['Email']

    if not recipient_email or pd.isna(recipient_email) or '@' not in recipient_email:
        print(f"Skipping {author_name} due to invalid email.")
        return

    message = MIMEMultipart()
    message["From"] = sender_email
    
    if test_email:
        print(f"TEST MODE: Sending email for {author_name} to {test_email} (original: {recipient_email})")
        message["To"] = test_email
    else:
        message["To"] = recipient_email

    message["Subject"] = subject
    message.attach(MIMEText(email_body, "plain"))

    mail_folder = "mail/"
    try:
        pdf_files = [f for f in os.listdir(mail_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"Warning: No PDF files found in {mail_folder} to attach.")

        for pdf_file in pdf_files:
            file_path = os.path.join(mail_folder, pdf_file)
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(file_path)}",
            )
            message.attach(part)
    except FileNotFoundError:
        print(f"Warning: The directory {mail_folder} was not found. Sending without attachments.")
    except Exception as e:
        print(f"An error occurred while attaching files: {e}")

    try:
        server.sendmail(sender_email, message["To"], message.as_string())
        print(f"Email sent to {author_name} at {message['To']}")
    except Exception as e:
        print(f"Failed to send email to {message['To']}: {e}")


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
                    top_parts = arg.split()
                    if not top_parts:
                        print("Usage: /top <number> [groups|schools|authors|companies]")
                        continue
                    try:
                        leaderboard_length = int(top_parts[0])
                        print(f"Leaderboard length set to {leaderboard_length}.")
                        category = 'all'
                        if len(top_parts) > 1:
                            category_arg = top_parts[1].lower()
                            if category_arg in ['groups', 'schools', 'authors', 'companies']:
                                category = category_arg
                            else:
                                print(f"Unknown category: {category_arg}. Showing all leaderboards.")
                        show_leaderboards(df, leaderboard_length, which=category)
                    except ValueError:
                        print("Invalid number for /top command. Please use an integer.")
                elif cmd == '/from':
                    if not arg:
                        print("Please specify an institution for the /from command.")
                        continue
                    institution = arg.strip('"\'')
                    show_authors_from(df, institution, leaderboard_length)
                elif cmd == '/findcontact':
                    if not arg:
                        print("Please specify an author\'s full name or email in quotes.")
                        print("Usage: /findcontact \"First Last\" or /findcontact \"user@example.com\"")
                        continue

                    search_term = arg.strip('"\'')

                    if '@' in search_term:
                        email_to_find = search_term.lower()
                        try:
                            contacts_df = pd.read_csv("contacts.csv")
                            contact_row = contacts_df[contacts_df['Email'].str.lower() == email_to_find]

                            if contact_row.empty:
                                print(f"Email '{email_to_find}' not found in contacts.csv.")
                                continue

                            contact_info = contact_row.iloc[0].to_dict()
                            author_name = contact_info['Author']

                            print("\n--- Contact Information (from contacts.csv) ---")
                            print(f"  Name: {contact_info.get('Author', 'N/A')}")
                            print(f"  Affiliation: {contact_info.get('Affiliation', 'N/A')}")
                            print(f"  Website: {contact_info.get('Website', 'N/A')}")
                            print(f"  LinkedIn: {contact_info.get('LinkedIn', 'N/A')}")
                            print(f"  Google Scholar: {contact_info.get('Google Scholar', 'N/A')}")
                            print(f"  Email: {contact_info.get('Email', 'N/A')}")

                        except FileNotFoundError:
                            print("Error: contacts.csv not found. Cannot search by email.")
                            print("Please run /getcontacts with the -save flag first.")
                            continue
                    else:
                        author_name = search_term
                        author_data = df[df['Author'].str.lower() == author_name.lower()]
                        if author_data.empty:
                            print(f"Author '{author_name}' not found in the database.")
                            continue

                        most_common_affiliation = author_data['Affiliation'].dropna().mode()
                        affiliation = most_common_affiliation[0] if not most_common_affiliation.empty else None
                        
                        await get_contacts([(author_name, affiliation)])

                    # print("\n--- Papers by this Author ---")
                    author_papers = df[df['Author'].str.lower() == author_name.lower()]
                    if author_papers.empty:
                        print("No papers found for this author in the database.")
                    else:
                        for _, paper in author_papers.iterrows():
                            print(f"  - [{paper['Year']}] {paper['Title']} ({paper['Conference']})")
                    print("\n" + "="*55 + "\n")

                elif cmd == '/findpaper':
                    if not arg:
                        print("Please specify a keyword to search for in paper titles.")
                        print("Usage: /findpaper \"keyword\"")
                        continue

                    keyword = arg.strip('"\'')
                    
                    matching_papers_df = df[df['Title'].str.contains(keyword, case=False, na=False)]

                    if matching_papers_df.empty:
                        print(f"No papers found with the keyword '{keyword}' in the title.")
                        continue

                    print(f"\n--- Papers containing '{keyword}' ---")
                    
                    grouped_by_paper = matching_papers_df.groupby(['Conference', 'Year', 'Title'])['Author'].apply(list).reset_index()

                    for _, paper in grouped_by_paper.iterrows():
                        print(f"\nTitle: {paper['Title']}")
                        print(f"  Conference: {paper['Conference']} ({paper['Year']})")
                        authors_str = ", ".join(paper['Author'])
                        print(f"  Authors: {authors_str}")
                        
                    print("\n" + "="*55 + "\n")

                elif cmd == '/getcontacts':
                    
                    arg_str = arg
                    save_to_csv = '-save' in arg_str
                    send_email_flag = '--send-email' in arg_str
                    filename = "contacts.csv"

                    k_match = re.match(r'^\s*(\d+)', arg_str)
                    if not k_match:
                        print("Invalid format. Use: /getcontacts <k> [\"institution1\"] [\"institution2\"]... [-save [filename.csv]] [--send-email]")
                        continue
                    k = int(k_match.group(1))
                    
                    institutions = re.findall(r'\"([^"]*?)\"', arg_str)
                    
                    if save_to_csv:
                        filename_match = re.search(r'-save\s+([a-zA-Z0-9_]+\.csv)', arg_str)
                        if filename_match:
                            filename = filename_match.group(1)

                    authors_info = []
                    if institutions:
                        for institution in institutions:
                            # print(f"\n--- Getting top {k} authors from {institution} ---")
                            inst_df = df[df['Affiliation'].str.contains(institution, case=False, na=False)]
                            if inst_df.empty:
                                print(f"No authors found for institution matching '{institution}'.")
                                continue
                            
                            top_authors_series = inst_df['Author'].value_counts().head(k)
                            for author_name in top_authors_series.index:
                                author_affiliations = inst_df[inst_df['Author'] == author_name]['Affiliation']
                                most_common_affiliation = author_affiliations.dropna().mode()
                                if not most_common_affiliation.empty:
                                    affiliation = most_common_affiliation[0]
                                else:
                                    affiliation = institution
                                authors_info.append((author_name, affiliation))
                    else:
                        authors_df = df
                        top_authors_series = authors_df['Author'].value_counts().head(k)
                        for author_name in top_authors_series.index:
                            author_affiliations = authors_df[authors_df['Author'] == author_name]['Affiliation']
                            most_common_affiliation = author_affiliations.dropna().mode()
                            if not most_common_affiliation.empty:
                                affiliation = most_common_affiliation[0]
                            else:
                                affiliation = None
                            authors_info.append((author_name, affiliation))

                    if not authors_info:
                        print("No authors found matching the criteria.")
                        continue
                    
                    contacts_list = await get_contacts(authors_info)

                    if send_email_flag and contacts_list:
                        print("--- Preparing to Send Emails ---")
                        try:
                            with open(args.email_template, 'r') as f:
                                lines = f.readlines()
                            subject_template = lines[0].strip() if lines else "Research Opportunities Inquiry"
                            email_body_template = "".join(lines[1:]) if len(lines) > 1 else ""
                        except FileNotFoundError:
                            print(f"Error: Email template file '{args.email_template}' not found. Cannot send emails.")
                            continue
                        
                        sender_email = input("Please enter your Gmail address: ")
                        password = getpass.getpass("Please enter your Gmail password or app password: ")
                        
                        context = ssl.create_default_context()
                        try:
                            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                server.login(sender_email, password)
                                print("Logged in successfully. Starting to send emails...")
                                for contact_info in contacts_list:
                                    await send_outreach_email(server, sender_email, contact_info, df, subject_template, email_body_template, args.prof, args.test)
                        except smtplib.SMTPAuthenticationError:
                            print("Failed to login. Please check your email and password.")
                            print("If you use 2-Step Verification, you may need to create an App Password.")
                        except Exception as e:
                            print(f"An error occurred while sending emails: {e}")


                    if save_to_csv and contacts_list:
                        contacts_df = pd.DataFrame(contacts_list)
                        contacts_df.to_csv(filename, index=False)
                        print(f"Contact information saved to {filename}")


                elif cmd == '/show':
                    if arg and arg in ['groups', 'schools', 'authors', 'companies']:
                        show_leaderboards(df, leaderboard_length, which=arg)
                    else:
                        show_leaderboards(df, leaderboard_length)
                elif cmd == '/help':
                    print("\nAvailable commands:")
                    print("  /show [groups|schools|authors|companies] - Display all or specific top leaderboards.")
                    print("  /top <number> [category] - Set leaderboard length and optionally show a specific category.")
                    print("  /from \"<institution>\"  - Show top authors from an institution.")
                    print("  /findcontact \"<name_or_email>\" - Get contact info and papers for a specific author.")
                    print("  /findpaper \"<keyword>\" - Find papers with a keyword in the title.")
                    print("  /getcontacts <k> [\"institution1\"] [\"institution2\"]... [-save [filename.csv]] [--send-email] - Scrape contact info and optionally save or email.")
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


async def outreach_mode(args):
    try:
        contacts_df = pd.read_csv(args.contacts_file)
    except FileNotFoundError:
        print(f"Error: Contacts file '{args.contacts_file}' not found.")
        return

    try:
        papers_df = pd.read_csv(args.output)
    except FileNotFoundError:
        print(f"Error: Papers file '{args.output}' not found. This is needed to find the most recent paper.")
        return

    try:
        with open(args.email_template, 'r') as f:
            lines = f.readlines()
        subject_template = lines[0].strip() if lines else "Research Opportunities Inquiry"
        email_body_template = "".join(lines[1:]) if len(lines) > 1 else ""
    except FileNotFoundError:
        print(f"Error: Email template file '{args.email_template}' not found.")
        return

    sender_email = input("Please enter your Gmail address: ")
    password = getpass.getpass("Please enter your Gmail password or app password: ")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            print("Logged in successfully. Starting to send emails...")

            for index, row in contacts_df.iterrows():
                contact_info = row.to_dict()
                await send_outreach_email(server, sender_email, contact_info, papers_df, subject_template, email_body_template, args.prof, args.test)

    except smtplib.SMTPAuthenticationError:
        print("Failed to login. Please check your email and password.")
        print("If you use 2-Step Verification, you may need to create an App Password.")
    except Exception as e:
        print(f"An error occurred while sending emails: {e}")

async def main():
    parser = argparse.ArgumentParser(
        description="Scrape and analyze paper data from ICML, NeurIPS, and ICLR."
    )
    parser.add_argument(
        "mode",
        choices=["scrape", "analyze", "outreach"],
        help="The mode to run the script in: 'scrape' to gather data, 'analyze' to view statistics, or 'outreach' to send emails."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="papers.csv",
        help="File to store data. Used as input for analysis. [Default: papers.csv]",
    )
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
    parser.add_argument(
        "--contacts-file",
        default="contacts.csv",
        help="CSV file with contact information for outreach mode. [Default: contacts.csv]",
    )
    parser.add_argument(
        "--email-template",
        default="mail/template.txt",
        help="Text file containing the email template for outreach mode. [Default: mail/template.txt]",
    )
    parser.add_argument(
        "--prof",
        action="store_true",
        help="Use 'Professor' as the title in the email salutation.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Send all emails to this address for testing purposes.",
    )
    
    args = parser.parse_args()

    if args.mode == 'scrape':
        if not args.years:
            parser.error("argument --years is required for mode 'scrape'")
        await scrape_mode(args)
    elif args.mode == 'analyze':
        await analyze_mode(args)
    elif args.mode == 'outreach':
        await outreach_mode(args)

if __name__ == "__main__":
    asyncio.run(main())
