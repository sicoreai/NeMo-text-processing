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
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money in Greek, e.g.
        "€50" -> money { integer_part: "πενήντα" currency: "ευρώ" }
        "€10,50" -> money { integer_part: "δέκα" currency: "ευρώ" fractional_part: "πενήντα" currency_minor: "λεπτά" }
        "50€" -> money { integer_part: "πενήντα" currency: "ευρώ" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph

        # Currency symbols and their names
        # Format: symbol -> singular_name (for amounts, Greek uses same form)
        currency_symbols = pynini.string_map([
            ("€", "ευρώ"),
            ("$", "δολάρια"),
            ("£", "λίρες"),
            ("¥", "γιεν"),
        ])

        # For Euro: cents are "λεπτά"
        currency_minor = pynini.string_map([
            ("€", "λεπτά"),
            ("$", "σεντς"),
            ("£", "πένες"),
        ])

        # Decimal separator (Greek uses comma)
        decimal_separator = pynini.union(",", ".")

        # Integer amount
        integer_amount = pynini.closure(NEMO_DIGIT, 1)
        integer_graph = integer_amount @ cardinal_graph

        # Fractional amount (cents) - always 2 digits
        fractional_amount = NEMO_DIGIT ** 2
        delete_leading_zero = pynini.cdrewrite(pynutil.delete("0"), "[BOS]", NEMO_DIGIT, NEMO_SIGMA)
        fractional_graph = fractional_amount @ delete_leading_zero @ cardinal_graph

        # Pattern 1: €50 (symbol before number)
        graph_symbol_before = (
            pynutil.insert("currency: \"")
            + currency_symbols
            + pynutil.insert("\" integer_part: \"")
            + integer_graph
            + pynutil.insert("\"")
        )

        # Pattern 2: 50€ (symbol after number)
        graph_symbol_after = (
            pynutil.insert("integer_part: \"")
            + integer_graph
            + pynutil.insert("\" currency: \"")
            + currency_symbols
            + pynutil.insert("\"")
        )

        # Pattern 3: €10,50 (with cents, symbol before)
        graph_with_cents_before = (
            pynutil.insert("currency: \"")
            + currency_symbols
            + pynutil.insert("\" integer_part: \"")
            + integer_graph
            + pynutil.insert("\"")
            + pynutil.delete(decimal_separator)
            + pynutil.insert(" fractional_part: \"")
            + fractional_graph
            + pynutil.insert("\" currency_minor: \"λεπτά\"")
        )

        # Pattern 4: 10,50€ (with cents, symbol after)
        graph_with_cents_after = (
            pynutil.insert("integer_part: \"")
            + integer_graph
            + pynutil.insert("\"")
            + pynutil.delete(decimal_separator)
            + pynutil.insert(" fractional_part: \"")
            + fractional_graph
            + pynutil.insert("\" currency: \"")
            + currency_symbols
            + pynutil.insert("\" currency_minor: \"λεπτά\"")
        )

        # Combine all patterns
        graph = (
            graph_with_cents_before
            | graph_with_cents_after
            | graph_symbol_before
            | graph_symbol_after
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
