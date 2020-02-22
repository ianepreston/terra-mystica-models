"""Download all the JSON files containing information on Terra Mystica games"""
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def data_download():
    base_url = "https://terra.snellman.net/data/events/"
    html = requests.get(base_url).text
    soup = BeautifulSoup(html, "html.parser")
    json_strings = [
        link.get("href")
        for link in soup.find_all("a", {"href": re.compile(r"\d{4}-\d{2}.json")})
    ]
    raw_folder = Path(__file__).resolve().parents[2] / "data" / "raw"
    assert raw_folder.exists()
    for json_string in json_strings:
        json_url = base_url + json_string
        json_path = raw_folder / json_string
        if not json_path.exists():
            with open(json_path, "w") as outfile:
                json.dump(requests.get(json_url).json(), outfile)
    return True


if __name__ == "__main__":
    data_download()
