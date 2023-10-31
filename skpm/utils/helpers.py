def flatten_list(l: list):
    return [item for sublist in l for item in sublist]

def unique_list(l: list):
    return list(set(l))