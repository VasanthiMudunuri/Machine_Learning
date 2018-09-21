from collections import namedtuple
from operator import itemgetter
from copy import deepcopy
 
class KDTree(object):
    def __init__(self, points, bounds):
        def split_nodes(split, nodes):
            if not nodes:
                return None
            nodes.sort(key=itemgetter(split))
            middle = len(nodes) // 2
            element_middle = nodes[middle]
            while middle + 1 < len(nodes) and nodes[middle + 1][split] == element_middle[split]:
                middle += 1
 
            s2 = (split + 1) % len(element_middle)  
            return Tree(element_middle, split, split_nodes(s2, nodes[:middle]),
                                    split_nodes(s2, nodes[middle + 1:]))
        self.node = split_nodes(0, points)
        self.bounds = bounds
class Tree(object):
    def __init__(self, median, split, left, right):
        self.median = median
        self.split = split
        self.left = left
        self.right = right
class searchMaxMin(object):
    def __init__(self, minimum, maximum):
        self.minimum, self.maximum = minimum, maximum
Result = namedtuple("Result", "nearest distance")
def find_nearest(tree, point):
    def Nearest_Neighbour(kdt, target,level, max_distance):
        if kdt is None:
            return Result([0.0] * 2, float("inf"))
        middle_element = kdt.split
        pivot = kdt.median
        left_branch = deepcopy(level)
        right_branch = deepcopy(level)
        left_branch.maximum[middle_element] = pivot[middle_element]
        right_branch.minimum[middle_element] = pivot[middle_element]
 
        if target[middle_element] <= pivot[middle_element]:
            nearer_kdt, nearer_level = kdt.left, left_branch
            farther_kdt, farther_level = kdt.right, right_branch
        else:
            nearer_kdt, nearer_level = kdt.right, right_branch
            farther_kdt, farther_level = kdt.left, left_branch
 
        point1 = Nearest_Neighbour(nearer_kdt, target, nearer_level, max_distance)
        nearest = point1.nearest
        distance = point1.distance
 
        if distance < max_distance:
            max_distance = distance
        d = (pivot[middle_element] - target[middle_element]) ** 2
        if d > max_distance:
            return Result(nearest, distance)
        d = sum((c1 - c2) ** 2 for c1, c2 in zip(pivot,target))
        if d < max_distance:
            nearest = pivot
            distance = d
            max_distance = distance
        point2 =Nearest_Neighbour(farther_kdt, target, farther_level, max_distance)
        if point2.distance < distance:
            nearest = point2.nearest
            distance = point2.distance
        return Result(nearest, distance)
    return Nearest_Neighbour(tree.node, point, tree.bounds, float("inf"))
if __name__ == "__main__":
    P = lambda *Points: list(Points)
    sample = KDTree([P(2, 3), P(5, 4), P(9, 6), P(4, 7), P(8, 1), P(7, 2)],searchMaxMin(P(0, 0), P(10, 10)))
    find_point=find_nearest(sample, P(7, 4)) 
    print("Sample Point:", P(7, 4))
    print("Nearest Point:", find_point.nearest)        
   

