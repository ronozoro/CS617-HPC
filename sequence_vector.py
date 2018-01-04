from functools import reduce

data = {
    '0': set(),
    '1': set(),
    '2': set(),
    '3': set('0'.split()),
    '4': set('1'.split()),
    '5': set('2'.split()),
    '6': set('2'.split()),
    '7': set('3 4'.split()),
    '8': set('4'.split()),
    '9': set('8 5 6'.split()),
}


def sequence_vector(data):
    for k, v in data.items():
        v.discard(k)  # Ignore self dependencies
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        yield ' '.join(sorted(ordered))
        data = {item: (dep - ordered) for item, dep in data.items()
                if item not in ordered}
    assert not data, "A cyclic dependency exists amongst %r" % data


x = 0
for i in sequence_vector(data):
    print('Sequence ' + str(x) + ' =', i)
    x += 1
