import sys
import os
sys.path.append(os.getcwd())

from misc.dictionary_extension import DocumentTermDictionary
from gensim.models import KeyedVectors

def verify_bridging(term_dict, query_term: str, term_of_interest: str, potential_bridging_terms: list):
    # Find all documents with ToI and query term (should have no connection)
    query_docs = set(term_dict.term2key_doc.get(query_term, [])[1])
    ToI_docs = set(term_dict.term2key_doc.get(term_of_interest, [])[1])

    # get documents with PBTs
    PBT_docs = {i: set(term_dict.term2key_doc.get(i, [])[1]) for i in potential_bridging_terms}

    
    # get first layer pairs
    query_PBT_docs = {key: query_docs.intersection(val) for key, val in PBT_docs.items()}
    ToI_PBT_docs = {key: ToI_docs.intersection(val) for key, val in PBT_docs.items()}
    # filter non overlap elements
    
    query_PBT_docs = {key: val for key, val in query_PBT_docs.items() if len(val) > 0}
    ToI_PBT_docs = {key: val for key, val in ToI_PBT_docs.items() if len(val) > 0}
    
    
    result_dict = {key: val for key, val in query_PBT_docs.items() if ToI_PBT_docs.get(key, None) is not None}
    
    
    # is_bridging_term = [True if len(ToI) > 0 and len(query) > 0 else False for ToI, query in zip(query_PBT_docs, ToI_PBT_docs)]
    return result_dict.keys()


def ident_potential_bridging_terms(model: KeyedVectors, query_term: str, term_of_interest: str):
    # lower_bound_sim = model.similarity(query_term, term_of_interest)
    PBT = set(model.closer_than(query_term, term_of_interest))
    PBT.update(set(model.closer_than(term_of_interest, query_term)))
    return PBT


if __name__ == "__main__":
    # query_term = "CTEPH"
    # term_dict = DocumentTermDictionary().load(path=os.path.join(base_dir, "term_dictionary.dict"))
    # # model = KeyedVectors.load(os.path.join(base_dir, skip_gram_CBOW, f"w2v_{end_date}.kv"), mmap="r")
    # model = KeyedVectors.load(os.path.join(base_dir, f"w2v_{end_date}.kv"), mmap="r")
    #
    # for ToI in terms_of_interest:
    #     PBT = ident_potential_bridging_terms(model=model, query_term=query_term, term_of_interest=ToI)
    #     bridging_terms = verify_bridging(term_dict=term_dict, query_term=query_term, term_of_interest=ToI, potential_bridging_terms=PBT)
    #     print(ToI, bridging_terms)
    
    pass
    
