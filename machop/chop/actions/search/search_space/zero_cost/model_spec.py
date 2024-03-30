import copy
import numpy as np

# Graphviz is optional and only required for visualization.
try:
    import graphviz
except ImportError:
    pass


class ModelSpec(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format="channels_last"):
        """Initialize the module spec."""
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("matrix must be square")
        if shape[0] != len(ops):
            raise ValueError("length of ops must match matrix dimensions")
        if not is_upper_triangular(matrix):
            raise ValueError("matrix must be upper triangular")

        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()

        self.data_format = data_format

    def _prune(self):
        """Prune the extraneous parts of the graph."""
        num_vertices = np.shape(self.original_matrix)[0]

        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output)
        )

        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    # def hash_spec(self, canonical_ops):
    #     """Computes the isomorphism-invariant graph hash of this spec."""
    #     labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
    #     return graph_util.hash_module(self.matrix, labeling)

    def visualize(self):
        """Creates a dot graph. Can be visualized in colab directly."""
        num_vertices = np.shape(self.matrix)[0]
        g = graphviz.Digraph()
        g.node(str(0), "input")
        for v in range(1, num_vertices - 1):
            g.node(str(v), self.ops[v])
        g.node(str(num_vertices - 1), "output")

        for src in range(num_vertices - 1):
            for dst in range(src + 1, num_vertices):
                if self.matrix[src, dst]:
                    g.edge(str(src), str(dst))
        return g


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False
    return True
