from graphix.gflow import find_gflow, find_pauliflow
from graphix.opengraph import OpenGraph
import networkx as nx

from graphix_og_generator.og_generator import (
    BlockComposer,
    remove_inputs,
    get_series_composition,
    get_grid_composition,
)
from graphix_og_generator.og_blocks import get_og_1, get_og_2



def get_flows_open_graph(
    og: OpenGraph,
) -> tuple[dict[int, list[int]] | None, dict[int, list[int]] | None]:
    """Compute the general and Pauli flows of the input open graph"""

    graph = og.inside
    iset = set(og.inputs)
    oset = set(og.outputs)
    meas_planes = {i: m.plane for i, m in og.measurements.items()}
    meas_angles = {i: m.angle for i, m in og.measurements.items()}

    gf, _ = find_gflow(graph=graph, iset=iset, oset=oset, meas_planes=meas_planes)
    pf, _ = find_pauliflow(
        graph=graph,
        iset=iset,
        oset=oset,
        meas_planes=meas_planes,
        meas_angles=meas_angles,
    )

    return gf, pf


class TestOGG:
    ogg = BlockComposer([get_og_1(), get_og_2()])

    # Test minimal block
    def test_minimal_block(self) -> None:
        for og in self.ogg.og_blocks:
            _, pf = get_flows_open_graph(og)
            assert pf is not None  # has pflow

    # Test parallel composition
    def test_parallel(self) -> None:
        n_blocks = [1, 3, 5]
        nodes_ref = [20, 38, 56]

        og_lst, nodes_lst = self.ogg.generate_og(n_blocks, merged_nodes_max=0)
        for og, n, n_ref in zip(og_lst, nodes_lst, nodes_ref):
            _, pf = get_flows_open_graph(og)
            assert pf is not None  # has pflow
            assert n == n_ref

    # Test series composition removing inputs
    def test_series(self) -> None:
        n_blocks = [1, 3, 5]
        nodes_ref = [19, 35, 51]

        og_lst, nodes_lst = self.ogg.generate_og(n_blocks, merged_nodes_max=1)
        for og, n, n_ref in zip(og_lst, nodes_lst, nodes_ref):
            og = remove_inputs(og, ni_max=1)
            _, pf = get_flows_open_graph(og)
            assert pf is not None  # has pflow
            assert n == n_ref

    # Test random composition removing inputs
    def test_random(self) -> None:
        n_blocks = [1, 3, 5, 10]
        nodes_ref = [20, 40, 60, 110]
        ni_max_vals = [10, 20, 30, 40]

        og_lst, nodes_lst = self.ogg.generate_og(n_blocks, rnd=True)
        for og, n, n_ref, ni_max in zip(og_lst, nodes_lst, nodes_ref, ni_max_vals):
            og = remove_inputs(og, ni_max=ni_max)
            _, pf = get_flows_open_graph(og)
            assert pf is not None  # has pflow
            assert n <= n_ref


class TestCompositionFunctions:
    og_minimal = get_og_1()

    # Test minimal block
    def test_minimal_block(self) -> None:
        gf, pf = get_flows_open_graph(self.og_minimal)

        assert gf is None  # does not have gflow
        assert pf is not None  # has pflow

    # Test series composition
    def test_series(self) -> None:
        n_reps = 4

        for n in range(n_reps):
            og = get_series_composition(self.og_minimal, n)
            if n == 0:
                assert og.isclose(self.og_minimal)

            gf, pf = get_flows_open_graph(self.og_minimal)
            assert gf is None  # does not have gflow
            assert pf is not None  # has pflow

    # Test grid composition

    ## Minimal block
    def test_grid_1(self) -> None:
        n_rows = 1
        n_layers = 1

        og = get_grid_composition(self.og_minimal, n_rows, n_layers)

        assert nx.is_isomorphic(og.inside, self.og_minimal.inside)

    ## Hard-coded non-trivial example 1
    def test_grid_2(self) -> None:
        n_rows = 2
        n_layers = 2

        og = get_grid_composition(self.og_minimal, n_rows, n_layers)

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
            (10, 12),
            (11, 14),
            (12, 13),
            (13, 14),
            (12, 15),
            (13, 16),
            (14, 17),
            (15, 16),
            (16, 17),
            (15, 18),
            (17, 19),
            (9, 20),
            (20, 21),
            (21, 22),
            (18, 22),
            (20, 23),
            (21, 24),
            (22, 25),
            (23, 24),
            (24, 25),
            (23, 26),
            (25, 27),
        ]

        assert nx.is_isomorphic(og.inside, nx.Graph(edges))
        assert len(og.inputs) == 4
        assert len(og.outputs) == 4

        gf, pf = get_flows_open_graph(self.og_minimal)
        assert gf is None  # does not have gflow
        assert pf is not None  # has pflow

    ## Hard-coded non-trivial example 2. This is test is necessary to show the need to reorder outputs when adding layer l0
    def test_grid_3(self) -> None:
        n_rows = 2
        n_layers = 3

        og = get_grid_composition(self.og_minimal, n_rows, n_layers)

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
            (10, 12),
            (11, 14),
            (12, 13),
            (13, 14),
            (12, 15),
            (13, 16),
            (14, 17),
            (15, 16),
            (16, 17),
            (15, 18),
            (17, 19),
            (9, 20),
            (20, 21),
            (21, 22),
            (18, 22),
            (20, 23),
            (21, 24),
            (22, 25),
            (23, 24),
            (24, 25),
            (23, 26),
            (25, 27),
            (8, 28),
            (28, 29),
            (29, 30),
            (28, 31),
            (29, 32),
            (30, 33),
            (26, 30),
            (31, 34),
            (31, 32),
            (32, 33),
            (33, 35),
            (27, 36),
            (36, 37),
            (37, 38),
            (36, 39),
            (37, 40),
            (38, 41),
            (19, 38),
            (39, 40),
            (40, 41),
            (39, 42),
            (41, 43),
        ]

        assert nx.is_isomorphic(og.inside, nx.Graph(edges))
        assert len(og.inputs) == 4
        assert len(og.outputs) == 4

        gf, pf = get_flows_open_graph(self.og_minimal)
        assert gf is None  # does not have gflow
        assert pf is not None  # has pflow

    ## Property tests
    def test_grid_4(self) -> None:
        examples = [
            (2, 1),
            (3, 1),
            (4, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (3, 2),
            (4, 2),
            (2, 4),
            (3, 3),
            (4, 4),
        ]

        for n_rows, n_layers in examples:
            og = get_grid_composition(self.og_minimal, n_rows, n_layers)

            if n_layers == 1:
                assert nx.number_connected_components(og.inside) == n_rows
            else:
                assert nx.number_connected_components(og.inside) == 1

            assert nx.is_planar(og.inside)

            def count_nodes() -> int:
                if n_layers == 1:
                    n_blocks = n_rows
                    n_nodes_merged = 0
                elif n_rows == 1:
                    n_blocks = n_layers
                    n_nodes_merged = 10 + 4 * (n_layers - 4)
                else:
                    n_blocks = n_layers // 2 * (2 * n_rows - 1) + n_layers % 2 * n_rows  # total number of minimal pf og
                    n_blocks_external = (
                        2 * n_rows if n_layers % 2 else 2 * n_rows - 1
                    )  # number of minimal pf og with inputs or outputs
                    n_blocks_internal = n_blocks - n_blocks_external
                    n_nodes_merged = 4 * n_blocks_internal + 2 * n_blocks_external
                    if n_layers % 2 == 0:
                        n_nodes_merged -= 2  # if number of layers is even, next-to-last layer has a two unmerged nodes.

                return self.og_minimal.inside.order() * n_blocks - n_nodes_merged / 2

            assert og.inside.order() == count_nodes()

            n_xputs = 3 if n_rows == 1 and n_layers > 1 else n_rows * 2
            assert len(og.inputs) == n_xputs
            assert len(og.outputs) == n_xputs

            gf, pf = get_flows_open_graph(self.og_minimal)
            assert gf is None  # does not have gflow
            assert pf is not None  # has pflow
