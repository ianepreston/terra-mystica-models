"""Download all the JSON files containing information on Terra Mystica games"""
import datetime as dt
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def _check_cutoff(json_string):
    """Check if a file was available when this analysis was first run
    
    Parameters
    ----------
    json_string: str
        Name of the json file to check
    """
    cutoff_date = dt.date(2020, 1, 1)
    json_date = dt.datetime.strptime(
        json_string.replace(".json", "-01"), "%Y-%m-%d"
    ).date()
    return json_date <= cutoff_date


def data_download(cutoff_date=True):
    """Download the JSON files from snelman
    
    Parameters
    ----------
    cutoff_date: bool, default True
        Only download files up to the date that was available when this analysis was
        originally done
    """
    base_url = "https://terra.snellman.net/data/events/"
    html = requests.get(base_url).text
    soup = BeautifulSoup(html, "html.parser")
    json_strings = [
        link.get("href")
        for link in soup.find_all("a", {"href": re.compile(r"\d{4}-\d{2}.json")})
    ]
    raw_folder = Path(__file__).resolve().parents[2] / "data" / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    for json_string in json_strings:
        json_url = base_url + json_string
        json_path = raw_folder / json_string
        if cutoff_date:
            download_condition = (not json_path.exists()) & _check_cutoff(json_string)
        else:
            download_condition = not json_path.exists()
        if download_condition:
            with open(json_path, "w") as outfile:
                json.dump(requests.get(json_url).json(), outfile)
    return True


if __name__ == "__main__":
    data_download()
