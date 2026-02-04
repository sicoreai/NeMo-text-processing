# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measurements, e.g.
        measure { cardinal { integer: "πέντε" } units: "χιλιόμετρα" } -> πέντε χιλιόμετρα

    Args:
        cardinal: CardinalFst verbalizer
        decimal: DecimalFst verbalizer
        fraction: FractionFst verbalizer (optional)
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst = None,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        cardinal_graph = cardinal.fst
        decimal_graph = decimal.fst

        # Extract cardinal/decimal/fraction content
        cardinal_content = (
            pynutil.delete("cardinal {")
            + delete_space
            + cardinal_graph
            + delete_space
            + pynutil.delete("}")
        )

        decimal_content = (
            pynutil.delete("decimal {")
            + delete_space
            + decimal_graph
            + delete_space
            + pynutil.delete("}")
        )

        # Units
        units = (
            delete_space
            + pynutil.delete("units:")
            + delete_space
            + pynutil.insert(" ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Cardinal + units
        graph_cardinal = cardinal_content + units

        # Decimal + units
        graph_decimal = decimal_content + units

        graph = graph_cardinal | graph_decimal

        if fraction is not None:
            fraction_graph = fraction.fst
            fraction_content = (
                pynutil.delete("fraction {")
                + delete_space
                + fraction_graph
                + delete_space
                + pynutil.delete("}")
            )
            graph_fraction = fraction_content + units
            graph |= graph_fraction

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
