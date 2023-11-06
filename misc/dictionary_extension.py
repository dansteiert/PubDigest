import pickle

class DocumentTermDictionary:
    def __init__(self):
        self.term2key_doc = dict()
        self.key2term = dict()
        self.doc2year = dict()
        self.current_id = 0
        
    def add_to_dict(self, term, doc_id, year):
        key_id = self.term2key_doc.get(term, None)
        if key_id is None:
            self.term2key_doc[term] = (self.current_id, [doc_id])
            self.current_id += 1
        else:
            if key_id[1][-1] == doc_id:
                pass
            else:
                doc_list = key_id[1]
                doc_list.append(doc_id)
                self.term2key_doc[term] = (key_id[0], doc_list)
        if self.doc2year.get(doc_id, None) is None:
            self.doc2year[doc_id] = year
            
    def build_key2term(self):
        self.key2term = {v[0]: (k, v[1]) for k, v in self.term2key}
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
            