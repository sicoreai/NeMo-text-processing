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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path, load_labels


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measurements in Greek, e.g.
        "5 km" -> measure { cardinal { integer: "πέντε" } units: "χιλιόμετρα" }
        "1 kg" -> measure { cardinal { integer: "ένα" } units: "κιλό" }
        "3,5 m" -> measure { decimal { integer_part: "τρία" fractional_part: "πέντε" } units: "μέτρα" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst (optional)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst = None,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.fst
        decimal_graph = decimal.fst

        # Load measurement units
        # Format: unit_symbol \t singular \t plural
        measurements_data = load_labels(get_abs_path("data/measures/measurements.tsv"))

        # Create mapping from symbol to singular/plural based on context
        # For simplicity, use plural form for values > 1
        unit_singular = []
        unit_plural = []
        for row in measurements_data:
            if len(row) >= 2:
                symbol = row[0]
                singular = row[1]
                plural = row[2] if len(row) > 2 else singular
                unit_singular.append((symbol, singular))
                unit_plural.append((symbol, plural))

        units_singular = pynini.string_map(unit_singular) if unit_singular else pynini.accep("")
        units_plural = pynini.string_map(unit_plural) if unit_plural else pynini.accep("")

        # For now, use plural as default (most common)
        units = units_plural | units_singular

        # Optional space between number and unit
        optional_space = pynini.closure(pynutil.delete(" "), 0, 1)

        # Cardinal measure: "5 km"
        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + cardinal_graph
            + pynutil.insert(" }")
            + optional_space
            + pynutil.insert(" units: \"")
            + units
            + pynutil.insert("\"")
        )

        # Decimal measure: "3,5 m"
        graph_decimal = (
            pynutil.insert("decimal { ")
            + decimal_graph
            + pynutil.insert(" }")
            + optional_space
            + pynutil.insert(" units: \"")
            + units
            + pynutil.insert("\"")
        )

        # Combine graphs
        graph = graph_cardinal | graph_decimal

        if fraction is not None:
            fraction_graph = fraction.fst
            graph_fraction = (
                pynutil.insert("fraction { ")
                + fraction_graph
                + pynutil.insert(" }")
                + optional_space
                + pynutil.insert(" units: \"")
                + units
                + pynutil.insert("\"")
            )
            graph |= graph_fraction

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
