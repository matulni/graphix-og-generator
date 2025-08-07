"""Open graph blocks with specific flow properties for composition."""

import networkx as nx
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph


def get_og_1() -> OpenGraph:
    """Return an open graph with that has Pauli flow but no gflow and equal number of outputs and inputs.

    The returned open graph has the following structure:

    [0]-2-5-(8)
        | |
        3-6
        | |
    [1]-4-7-(9)

    Adapted from Fig. 7 in D. E. Browne et al 2007 New J. Phys. 9 250.
    """

    edges = [
        (0, 2),
        (1, 4),
        (2, 3),
        (3, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 6),
        (6, 7),
        (5, 8),
        (7, 9),
    ]

    graph = nx.Graph(edges)

    inputs = [0, 1]
    outputs = [8, 9]
    measurements = {i: Measurement(0, Plane.XY) for i in range(8)}

    og = OpenGraph(graph, measurements, inputs, outputs)

    return og


def get_og_2() -> OpenGraph:
    """Return an open graph with Pauli flow and equal number of outputs and inputs.

    The returned graph has the following structure:

    [0]-2-4-(6)
        | |
    [1]-3-5-(7)
    """
    graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
    inputs = [0, 1]
    outputs = [6, 7]
    meas = {
        0: Measurement(0.1, Plane.XY),  # XY
        1: Measurement(0.1, Plane.XY),  # XY
        2: Measurement(0.1, Plane.XY),  # X
        3: Measurement(0.1, Plane.XY),  # XY
        4: Measurement(0.0, Plane.XY),  # X
        5: Measurement(0.5, Plane.YZ),  # Y
    }
    return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)
