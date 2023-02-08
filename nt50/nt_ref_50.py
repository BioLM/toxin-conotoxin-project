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


#url = 'https://rest.uniprot.org/uniref/search?fields=id%2Cname%2Ctypes%2Ccount%2Corganism%2Clength%2Cidentity%2Csequence&format=tsv&query=%28%28length%3A%5B4%20TO%20500%5D%29%20NOT%20toxin%20NOT%20virus%29%20AND%20%28identity%3A0.5%29&size=500'
url='https://rest.uniprot.org/uniref/search?format=tsv&fields=id,name,types,count,organism,length,identity,sequence&query=((length:%5B4%20TO%20500%5D)%20NOT%20toxin%20NOT%20virus)%20AND%20(identity:0.5)&cursor=1z1z1n7bxa4c2wvmfitbh36tztkkmuyalpc56h1v3tkom1n8w0y80mv7j6nlglr2lkon5d&size=500'
#progress = 0
progress = 41002000
for batch, total, link in get_batch(url):
    lines = batch.text.splitlines()
    with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniRef/UniRef_NotToxins/50level/ntref50-4to500#'+str(progress)+'.tsv', 'w') as f:
        if not progress:
            print(lines[0], file=f)
        for line in lines[1:]:
            print(line, file=f)
        progress += len(lines[1:])
        with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniRef/UniRef_NotToxins/50level/ntref50-4to500.txt', 'w') as f2:
            print([f'{progress} / {total}',link],file=f2)
        with open('/home/chance/biolmtoxin/toxin-conotoxin-project/UniRef/UniRef_NotToxins/50level/ntref50-4to500links.txt', 'w') as f3:
            print('\n'+link,file=f3)