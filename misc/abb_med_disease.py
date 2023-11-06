
def gen_abb_med_disease(abb: bool, med: bool, disease: bool):
    return '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ("_disease" if disease else "")))