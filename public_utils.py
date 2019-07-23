

def deep_flatten(l):
    if l is None:
        return []
    if type(l) is str:
        return [l]
    res=[]
    for i in l:
        res.extend(deep_flatten(i))
    return res