from decimal import Decimal

def _norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, Decimal)):
        return round(float(v), 4)
    if isinstance(v, str):
        return v.strip().casefold()
    return v

def _row(r):
    # ignore column order and names within a row
    return tuple(sorted((_norm(v) for v in r), key=repr))

def results_match(gold, pred, ordered=False):
    g = [_row(r) for r in gold]
    p = [_row(r) for r in pred]
    if not ordered:
        g.sort(key=repr)
        p.sort(key=repr)
    return g == p
