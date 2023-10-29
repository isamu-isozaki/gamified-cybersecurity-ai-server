from bs4 import BeautifulSoup
import bs4
from urllib.request import urlopen
import urllib
import os
import pandas as pd
import gc

def parse_red_team_document(page):
    page_content = page.findAll("main", {"class": ["r-1oszu61"]})[0].get_text()
    return page_content
