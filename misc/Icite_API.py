import requests
import datetime

# TODO: ciations does not get all citations!
def get_citations(pmids, fields_to_return: list = ["pmid", "citation_count", "is_research_article", "is_clinical"],
                  chunk_size:int = 500):
    # print(datetime.datetime.now(), "iCite progress: ", end="")
    base_url = "https://icite.od.nih.gov/api/pubs"
    result_dict_list = []
    
    chunks = (len(pmids) - 1) // chunk_size + 1
    for i in range(chunks):
        batch = pmids[i * chunk_size:(i + 1) * chunk_size]
        
        # see https://icite.od.nih.gov/api for more information
        parameters = {}
        parameters["pmids"] = ",".join([str(i) for i in batch])
        
        response = requests.get(url=base_url, params=parameters)
        pubs = response.json()
        if isinstance(fields_to_return, list):
            result_dict_list.extend(
                [{k: v for k, v in d.items() if k in fields_to_return and v is not None} for d in pubs.get("data", [])])
        else:
            result_dict_list.extend(pubs.get("data", []))
        # print("#", end="")
    # print("")

    return result_dict_list

def get_citation_single(pmid,
                        fields_to_return: list = ["pmid", "citation_count", "is_research_article", "is_clinical"]):
    base_url = "https://icite.od.nih.gov/api/pubs"

    parameters = {"pmids": str(pmid)}
    response = requests.get(url=base_url, params=parameters)
    pub = response.json()
    if isinstance(fields_to_return, list):
        result_dict ={k: v for k, v in pub.items() if k in fields_to_return and v is not None}
    else:
        result_dict_list = pub.get("data", [])
        if len(result_dict_list) > 0:
            result_dict = result_dict_list[0]
        else:
            return {}
    result_dict = {f"icite_{k}": v for k, v in result_dict.items()}
    return result_dict
