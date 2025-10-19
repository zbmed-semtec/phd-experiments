import csv
from Bio.Entrez import efetch, read


def fetch_pubmed_articles(pmid):
    handle = efetch(db='pubmed', id=pmid, retmode='xml')
    xml_data = read(handle)
    try:
        title = xml_data['PubmedArticle'][0]["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = xml_data['PubmedArticle'][0]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
        return title, abstract
    except IndexError:
        return None
    
PMIDS = [28600191, 19243676, 10793294, 24463521, 25984610]

with open("data/pubmed_papers.tsv", "w", newline='', encoding="utf-8") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t")
    writer.writerow(["PMID", "Title", "Abstract"])

    for pmid in PMIDS:
        title, abstract = fetch_pubmed_articles(pmid)
        if title and abstract:
            writer.writerow([pmid, title, abstract])
        else:
            print(f"PMID: {pmid} not found")


# 28600191: drug disease relations
# 19243676: clinical note
# 10793294: gene protein interactions
# 24463521: drug discovery
# 25984610: multiple entities

