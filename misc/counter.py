def count_dict(l: list):
    count_dict = {}
    for i in l:
        count_dict[i] = count_dict.get(i, 0) + 1
    return count_dict


def count_dict_two_level(l: list):
    count_dict = {}
    for i in l:
        for j in i:
            count_dict[j] = count_dict.get(j, 0) + 1
    return count_dict


def count_dict_three_level(l: list):
    count_dict = {}
    for i in l:
        for j in i:
            for k in j:
                count_dict[k] = count_dict.get(k, 0) + 1
    return count_dict


def get_most_occuring_element(x):
    counts = count_dict(x)

    if len(counts.items()) > 0:
        return max(counts, key=counts.get)
    else:
        return None


