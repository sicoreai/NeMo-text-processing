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
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions in Greek, e.g.
        "1/2" -> fraction { numerator: "ένα" denominator: "δύο" }
        "3/4" -> fraction { numerator: "τρία" denominator: "τέσσερα" }

    Common fractions have special names in Greek:
        1/2 = μισό
        1/4 = τέταρτο

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph

        # Fraction separator
        fraction_separator = pynini.union("/", "⁄")

        # Special fractions
        special_fractions = pynini.string_map([
            ("1/2", "μισό"),
            ("1/4", "ένα τέταρτο"),
            ("3/4", "τρία τέταρτα"),
        ])

        # Numerator
        numerator = pynini.closure(NEMO_DIGIT, 1) @ cardinal_graph
        numerator_field = pynutil.insert("numerator: \"") + numerator + pynutil.insert("\"")

        # Denominator
        denominator = pynini.closure(NEMO_DIGIT, 1) @ cardinal_graph
        denominator_field = pynutil.insert(" denominator: \"") + denominator + pynutil.insert("\"")

        # Standard fraction: numerator/denominator
        graph_standard = (
            numerator_field
            + pynutil.delete(fraction_separator)
            + denominator_field
        )

        # Special fractions (resolved directly)
        graph_special = (
            pynutil.insert("numerator: \"")
            + special_fractions
            + pynutil.insert("\" denominator: \"\"")
        )

        # Integer + fraction: "2 1/2"
        integer_field = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_mixed = (
            integer_field
            + pynutil.delete(" ")
            + pynutil.insert(" ")
            + numerator_field
            + pynutil.delete(fraction_separator)
            + denominator_field
        )

        graph = graph_standard | graph_mixed

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
