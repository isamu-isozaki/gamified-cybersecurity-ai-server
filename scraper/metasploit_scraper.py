from bs4 import BeautifulSoup
import bs4
from urllib.request import urlopen
import urllib
import os
import pandas as pd
import gc

def parse_metasploit_document(page):
    page_content = page.findAll("div", {"id": "main-content"})[0].get_text()
    return page_content