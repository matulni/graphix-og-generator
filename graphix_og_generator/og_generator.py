from copy import copy, deepcopy
import random
from itertools import cycle
from typing import Iterable

from graphix.opengraph import OpenGraph

from typing_extensions import deprecated


class BlockComposer:
    """A class for constructing open graphs by composing a set of a minimal building blocks. The resulting open graph preserves the flow properties of the building blocks.

    Attributes
    ----------
        og_blocks (list[OpenGraph]) : Minimal building blocks employed in the composition.
    """

    def __init__(self, blocks: Iterable[OpenGraph]) -> None:
        self.og_blocks = list(blocks)

    def generate_og(
        self,
        n_blocks: Iterable[int],
        merged_nodes_max: int = -1,
        ni_max_vals: Iterable[int] | None = None,
        rnd: bool = False,
        seed: int = 42,
    ) -> tuple[list[OpenGraph], list[int]]:
        r"""Construct a list of open graphs by iteratively composing the minimal blocks in `self.og_blocks`.

        Parameters
        ----------
        n_blocks : Iterable[int]
            Number of blocks of the constructed open graphs. Must be larger or equal than 1.
        merged_nodes_max: int
            Maximum number of merged nodes at each step. By default (`merged_nodes_max = -1`) all inputs or outputs (whichever is smaller) of the composed open graphs are merged.
        ni_max_vals : Iterable[int] | None (optional)
            Maximum number of inputs in the returned open graphs. By default (`ni_max = None`) no maximum is enforced.
        rnd : bool (optional)
            Flag for selecting an stochastic construction of the open graph (see Notes for more details). It defaults to `False`.
        seed : int (optional)
            Seed for the random number generator. It defaults to 42.

        Returns
        -------
        list[OpenGraph]
            List of composed open graphs.
        list[int]
            Number of nodes of the returned open graphs.

        Notes
        -----
        - The first iteration of the construction process consists of selecting two open graphs in `self.og_blocks` (denoted `og1` and `og2`) and merging a subset of outputs of `og1` with a subset of inputs of `og2`. In subsequent steps of the iteration, the resulting open graph is assigned to `og1` and `og2` is always selected from `self.og_blocks`.

        - We perform `n = max(n_blocks)` iterations. The intermediate open graphs with :math:`n \in``n_blocks` blocks are stored and returned in ascending `n` order.

        - The maximum number of nodes merged at each step (`merged_nodes_max`) must be smaller or equal than `min(len(og1.outputs), min(len(og2.inputs))`. If zero, open graphs are composed in parallel.

        - A finite maximum of inputs is achieved by removing `len(og1.inputs) - ni_max` nodes from the input nodes set of the composed open graph at the end of the iteration process. The composed open graph after this operation perserves the flow properties.

        - If `rnd == False`:
            - `og2` is selected by cycling over `self.og_blocks`
            - The first `merged_nodes_max` are merged.
            - The first `delta_noni` inputs are removed.
        - If `random == True`:
            - `og2` is selected randomly from `self.og_blocks`
            - The amount of merged nodes is (uniformly) picked from (0, `merged_nodes_max`). The merged nodes are selected randomly.
            - The removed `delta_noni` inputs are selected randomly.
        """

        n_max = max(n_blocks)
        n_blocks_set = set(n_blocks)
        if rnd:
            random.seed(seed)

        og_lst: list[OpenGraph] = []
        og_nodes: list[int] = []

        def get_mapping(merged_nodes_max) -> dict[int, int]:
            min_io = min(len(og1.outputs), len(og2.inputs))

            if merged_nodes_max > min_io or merged_nodes_max < 0:
                merged_nodes_max = min_io

            # NOTE: In future versions `inputs` and `outputs` of `OpenGraph` will be (unordered) sets which don't support indexing.
            ins = og2.inputs[:merged_nodes_max] if not rnd else random.choices(og2.inputs, k=merged_nodes_max)
            outs = og1.outputs[:merged_nodes_max] if not rnd else random.choices(og1.outputs, k=merged_nodes_max)

            return dict(zip(ins, outs))

        og1 = self.og_blocks[0] if not rnd else random.choice(self.og_blocks)

        for n, og2 in enumerate(cycle(self.og_blocks), start=1):
            if n == n_max:
                break
            mapping = get_mapping(merged_nodes_max)
            og1, _ = og1.compose(og2, mapping)
            if n in n_blocks_set:
                og_lst.append(og1)
                og_nodes.append(og1.inside.order())

        if ni_max_vals:
            for og, ni_max in zip(og_lst, ni_max_vals):
                ni_remove = max(0, len(og.inputs) - ni_max)
                ins = og.inputs[ni_remove:] if not rnd else random.choices(og.inputs, k=ni_remove)
                for i in ins:
                    og.inputs.remove(i)

        return og_lst, og_nodes


#### Deprecated functions
# Even though the behavior of `get_series_composition` cannot be replicated with `BlockComposer.generate_og`, the function is deprecated as it is unnecessarily complicated for the purpose of constructing open graphs which preserve the flow properties.


@deprecated("Use `BlockComposer.generate_og` instead")
def get_series_composition(og: OpenGraph, n: int) -> OpenGraph:
    """Construct an open graph by composing in series n copies of a minimal block.

    Example: n = 3
      _ _  _ _  _ _
    -|   ||   ||   |-
    -|_ _||_ _||_ _|-

    Parameters
    ----------
    og0 : OpenGraph
        Minimal block
    n : int
        Number of copies

    Returns
    -------
    OpenGraph
        Composed Open Graph
    """
    og_aux = deepcopy(og)

    for _ in range(n):
        mapping = dict(zip(og_aux.inputs, og.outputs))
        og, _ = og.compose(og_aux, mapping)

    return og


@deprecated("Use `BlockComposer.generate_og` instead")
def get_grid_composition(og: OpenGraph, n_rows: int, n_layers: int) -> OpenGraph:
    """Construct an open graph by composing a minimal block in a brick-wall structure.

    - If n_layers = 1, the returned open graph consists of n_rows blocks composed in parallel.

    Example: n_rows = 2, n_layers = 3
      _ _       _ _
    -|   | _ _ |   |-
    -|_ _||   ||_ _|-
    -|‾ ‾||_ _||‾ ‾|-
    -|_ _|     |_ _|-

    Parameters
    ----------
    og : OpenGraph
        Minimal block
    n_rows : int
        Number of blocks in first layer and succesive odd layers. Must be equal or larger than 1.
    n_layers : int
        Number of layers. Must be equal or larger than 1.

    Returns
    -------
    og: OpenGraph
    """

    if n_layers < 1:
        raise ValueError("Number of layers  must be larger than 1")
    if n_rows < 1:
        raise ValueError("Number of rows  must be larger than 1")

    def get_layers(n_rows: int) -> tuple[OpenGraph, OpenGraph]:
        """Compose blocks in parallel to obtain the alternating layers."""

        og0 = og
        og1 = deepcopy(og)
        og_aux = deepcopy(og)

        for n in range(1, n_rows):
            shift = og0.inside.order()
            mapping = {node: i for i, node in enumerate(og_aux.inputs + og_aux.outputs, start=shift)}
            og0, _ = og0.compose(og_aux, mapping)

            if (
                n < n_rows - 1
            ):  # l1 always has one block less than l0, unless n_rows = 1, in which case both l0 and l1 have 1 block.
                og1, _ = og1.compose(og_aux, mapping)

        return og0, og1

    def add_l0(og: OpenGraph) -> OpenGraph:
        if len(og.outputs) == 3:
            outputs = og.outputs[:-1]
        else:
            # we move the second output of og to the end
            outputs = copy(og.outputs)
            outputs.append(outputs[1])
            outputs.remove(outputs[1])

        mapping = {i: o for (i, o) in zip(l0.inputs, outputs)}
        og, _ = og.compose(l0, mapping)

        return og

    def add_l1(og: OpenGraph) -> OpenGraph:
        outputs = og.outputs[1:-1] if len(og.outputs) > 3 else og.outputs[1:]
        mapping = {i: o for (i, o) in zip(l1.inputs, outputs)}
        og, _ = og.compose(l1, mapping)

        return og

    l0, l1 = get_layers(n_rows)
    og = deepcopy(l0)  # initial layer

    for i, add_layer in enumerate(cycle([add_l1, add_l0])):
        if i == n_layers - 1:
            break
        og = add_layer(og)

    return og
