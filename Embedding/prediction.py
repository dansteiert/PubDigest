from gensim.models import KeyedVectors



def prediction(model: KeyedVectors, base_term: str, top_prediction_terms: int):
    try:
        return model.most_similar(positive=[base_term], topn=top_prediction_terms)
    except:
        return None


