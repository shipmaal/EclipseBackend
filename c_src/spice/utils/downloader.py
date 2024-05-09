import os
from bs4 import BeautifulSoup
from functools import lru_cache
from requests import Session

class Downloader:
    def __init__(self, url):
        self.session = Session()
        self.url = url

    def list_files(self):
        response = self.session.get(self.url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')
            files = []
            for link in links:
                href = link.get('href')
                if href and href.endswith('.bsp') and not href.startswith('?'):
                    files.append(href)
            return files
        else:
            print(f"Failed to retrieve contents from {self.url}")
            return []

    def download_file(self, filename):
        with self.session.get(self.url + filename, stream=True) as response:
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename}")

    @lru_cache(maxsize=None)
    def get_cached_files(self):
        return self.list_files()

    def ensure_file(self, filename):
        if os.path.exists(filename):
            print(f"{filename} already exists.")
            return
        try:
            files = self.get_cached_files()
            if filename in files:
                self.download_file(filename)
            else:
                print(f"{filename} not found in the directory listing.")
        except Exception as e:
            print(f"An error occurred: {e}")

