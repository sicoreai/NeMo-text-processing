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
    insert_space,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals in Greek, e.g.
        "3,14" -> decimal { integer_part: "τρία" fractional_part: "δεκατέσσερα" }
        "0,5" -> decimal { integer_part: "μηδέν" fractional_part: "πέντε" }

    Greek uses comma as decimal separator (not period).

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph

        # Greek uses comma as decimal separator
        decimal_separator = pynini.union(",", ".")

        # Integer part
        integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        # Fractional part - read digit by digit for decimals
        # Load digit names
        zero = pynini.invert(pynini.string_file(get_abs_path("data/number/zero.tsv")))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv")))
        digit_with_zero = digit | zero

        # Read fractional digits one by one
        single_digit = digit_with_zero

        # For the fractional part, read each digit separately
        fractional_digits = pynini.closure(single_digit + insert_space, 0) + single_digit

        # Alternative: read as a number (e.g., 14 -> δεκατέσσερα)
        fractional_as_number = cardinal_graph

        # Use digit-by-digit reading as primary for decimals
        fractional_part = (
            pynutil.insert(" fractional_part: \"")
            + fractional_digits
            + pynutil.insert("\"")
        )

        # Build decimal graph
        # Pattern: integer_part + comma + fractional_part
        graph_decimal = (
            integer_part
            + pynutil.delete(decimal_separator)
            + fractional_part
        )

        # Also allow decimals without integer part (e.g., ",5" -> "μηδέν κόμμα πέντε")
        graph_decimal_no_integer = (
            pynutil.insert("integer_part: \"μηδέν\"")
            + pynutil.delete(decimal_separator)
            + fractional_part
        )

        graph = graph_decimal | graph_decimal_no_integer

        # Quantity suffix for large decimals (e.g., "3,5 εκατομμύρια")
        quantity = pynini.string_map([
            ("χιλιάδες", "χιλιάδες"),
            ("εκατομμύρια", "εκατομμύρια"),
            ("δισεκατομμύρια", "δισεκατομμύρια"),
        ])
        optional_quantity = pynini.closure(
            pynutil.insert(" quantity: \"")
            + pynutil.delete(" ")
            + quantity
            + pynutil.insert("\""),
            0,
            1,
        )

        graph += optional_quantity

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
