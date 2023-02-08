import re
import requests
from requests.adapters import HTTPAdapter, Retry

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        nextlink=get_next_link(response.headers)
        yield response, total, nextlink
        batch_url = get_next_link(response.headers)


url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Cid%2Cprotein_name%2Corganism_name%2Clength%2Csequence&format=tsv&query=%28%28length%3A%5B4%20TO%20500%5D%29%20NOT%20toxin%20NOT%20virus%29&size=500'
progress = 0
for batch, total, link in get_batch(url):
    lines = batch.text.splitlines()
    with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniprotKB/Uniprot_NotToxins/All/ntprot-4to500#'+str(progress)+'.tsv', 'w') as f:
        if not progress:
            print(lines[0], file=f)
        for line in lines[1:]:
            print(line, file=f)
        progress += len(lines[1:])
        with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniprotKB/Uniprot_NotToxins/All/ntprot-4to500.txt', 'w') as f2:
            print([f'{progress} / {total}',link],file=f2)
        with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniprotKB/Uniprot_NotToxins/All/ntprot-4to500links.txt', 'w') as f3:
            print('\n'+link,file=f3)