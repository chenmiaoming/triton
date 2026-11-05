"""Deterministic balanced orders, frozen before any performance sample."""


def rows(tags, invocation):
    n = len(tags)
    assert 1 <= n <= 4
    # Williams order for even n; cyclic forward/reverse for odd n.
    pattern = [0]
    for i in range(1,n):
        pattern.append((i+1)//2 if i%2 else n-i//2)
    forward = [[tags[(p+offset)%n] for p in pattern] for offset in range(n)]
    orders = forward+[list(reversed(r)) for r in forward]
    offset = (invocation-1)%len(orders)
    return orders[offset:]+orders[:offset]


def schedule(groups, invocation):
    orders = {key:rows(value,invocation) for key,value in groups.items()}
    keys = sorted(groups)
    visits = []
    for fraction in (1,16):
        for r in range(max(map(len,orders.values()))):
            offset = (invocation-1+r)%len(keys)
            for key in keys[offset:]+keys[:offset]:
                if r < len(orders[key]):
                    for tag in orders[key][r]:
                        visits.append({'group':key,'round':r+1,'fraction':fraction,'tag':tag})
    return visits
