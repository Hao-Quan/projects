import sys

sys.path.extend(['../'])
from graph import tools

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 1), (6, 5), (7, 6),
                    (8, 1), (9, 8), (10, 9), (11, 1), (12, 11), (13, 12),
                    (14, 0), (15, 0), (16, 14), (17, 15)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

bone_weights = [[1, 1, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10,0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10],
                [10, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10],
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
                ]

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        #self.A = tools.normalize_adjacency_matrix(self.A_binary)
        self.A = tools.normalize_adjacency_matrix(self.A_binary_with_I)
        self.bone_weights = bone_weights

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
