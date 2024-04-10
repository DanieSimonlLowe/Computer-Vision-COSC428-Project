def get_m_and_c(p1, p2):
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    c = p1[1] - m * p1[0]

    return m, c


def intersects(p1, p2):
    m, c = get_m_and_c(p1, p2)

    part = (m ** 2 + 1)
    if part == 0:
        Exception('can\'t divide by 0 value')

    value = part * 9 - c ** 2
    return value >= 0


points1 = [
    ('a', (10, 5)),
    ('b', (10, 10)),
    ('c', (5, 10)),
    ('d', (0, 10)),
    ('e', (-5, 10)),
]


points2 = [
    ('A', (-10, 5)),
    ('B', (-10, 0)),
    ('C', (-5, 0)),
    ('D', (0, 0)),
    ('E', (5, 0)),
]

for label1, point1 in points1:
    for label2, point2 in points2:
        try:
            if intersects(point1, point2):
                print('{} and {} intersect'.format(label1, label2))
        except Exception as e:
            print(e)