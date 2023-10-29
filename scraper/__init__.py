from bs4 import BeautifulSoup
import bs4
from urllib.request import urlopen, Request
import urllib
import os
import pandas as pd
import gc
from urllib.parse import urljoin
from scraper.hack_tricks_scraper import parse_hack_tricks_document
from scraper.metasploit_scraper import parse_metasploit_document
from scraper.mitre_scraper import parse_mitre_document
from scraper.payload_all_things_scraper import parse_payload_all_things_document
from scraper.red_team_scraper import parse_red_team_document
from tqdm.auto import tqdm
import pickle

def parse_document(page):
    page_content = page.get_text()
    return page_content
def scrape_dataset(base_url="https://attack.mitre.org/", allowed_domains=["https://attack.mitre.org/"], depth=10, keep_to_allowed=True, output_dir="data/raw", subtask="mitre", callback=parse_document, save_frequency=100):
    """
    General scraper. The main idea behind this is to collect urls of a given website
    and then gather the relevant text into txt files ideally without html. This will be a bfs

    Args:
        base_url (str, optional): _description_. Defaults to "https://attack.mitre.org/".
    """
    current_state_path = os.path.join(output_dir, f'{subtask}.pkl')
    if os.path.exists(current_state_path):
        with open(current_state_path, 'rb') as f:
            current_state = pickle.load(f)
            current_urls = current_state["current_urls"]
            already_explored = current_state["already_explored"]
            resume_j = current_state["j"]
            resume_depth = current_state["curr_depth"]
            next_urls = current_state["next_urls"]
        print(f"Resuming from depth {resume_depth} at j {resume_j}")
    else:
        already_explored = {base_url}
        current_urls = [base_url]
        resume_depth = -1
        resume_j = -1
        next_urls = []

    j = 0
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, subtask)
    os.makedirs(output_dir, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    for curr_depth in range(depth):
        if curr_depth < resume_depth:
            continue
        for current_url in tqdm(current_urls):
            if j <= resume_j:
                j += 1
                continue
            try:
                request = Request(url=current_url, headers=headers)
                page = urlopen(request)
            except Exception as e:
                print(e)
                continue
            page = BeautifulSoup(page,"html.parser")

            page_path = os.path.join(output_dir, str(j)+'.txt')
            if not os.path.exists(page_path):
                try:
                    text = callback(page)
                    with open(page_path, "w+") as f:
                        f.write(text)
                except Exception as e:
                    text = ""
            if (j+1) % save_frequency == 0:
                output = {
                    "current_urls": current_urls,
                    "already_explored": already_explored,
                    "j": j,
                    "curr_depth": curr_depth,
                    "next_urls": next_urls
                }
                with open(current_state_path, 'wb') as f:
                    pickle.dump(output, f)
            j+=1
            links = page.findAll("a")
            for link in links:
                next_link = link.get('href')
                if next_link is None:
                    continue
                if not next_link.startswith("http"):
                    next_link = urljoin(base_url, next_link)
                if next_link in already_explored:
                    continue
                if "#" in next_link:
                    continue
                starts_with_allowed = False
                for allowed_domain in allowed_domains:
                    if next_link.startswith(allowed_domain):
                        starts_with_allowed=True
                        break
                if not starts_with_allowed and keep_to_allowed:
                    continue
                already_explored.add(next_link)
                next_urls.append(next_link)
        current_urls = next_urls
        print(f"Parsed {len(already_explored)} links")
        print(f"Next urls are {current_urls}")
        next_urls = []

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    skip_index = 0
    end_index = 100
    parsers = [parse_hack_tricks_document, parse_metasploit_document, parse_mitre_document, parse_payload_all_things_document, parse_red_team_document]
    db_names = ["hack_tricks", "metasploit", "mitre", "payload_all_things", "red_team"]
    base_urls = ["https://book.hacktricks.xyz", "https://docs.metasploit.com/", "https://attack.mitre.org/", "https://swisskyrepo.github.io/PayloadsAllTheThings/", "https://www.ired.team/"]

    for (base_url, parser, db_name) in zip(base_urls[skip_index:end_index], parsers[skip_index:end_index], db_names[skip_index:end_index]):
        print(f"Scraping {db_name}")
        scrape_dataset(base_url, allowed_domains=[base_url], keep_to_allowed=True, subtask=db_name, callback=parser)