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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "δέκα" currency: "ευρώ" fractional_part: "πενήντα" currency_minor: "λεπτά" }
        -> δέκα ευρώ και πενήντα λεπτά

    Args:
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # Currency
        currency = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Integer part
        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Fractional part (cents)
        fractional_part = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Currency minor (λεπτά)
        currency_minor = (
            pynutil.delete("currency_minor:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Graph for: integer currency [και fractional currency_minor]
        # Handle different orderings of fields

        # Pattern: currency integer_part
        graph_currency_first = (
            currency
            + pynutil.insert(" ")
            + delete_space
            + integer_part
        )

        # Pattern: integer_part currency
        graph_integer_first = (
            integer_part
            + pynutil.insert(" ")
            + delete_space
            + currency
        )

        # Pattern with cents: currency integer_part fractional_part currency_minor
        graph_with_cents = (
            currency
            + pynutil.insert(" ")
            + delete_space
            + integer_part
            + delete_space
            + pynutil.insert(" και ")
            + fractional_part
            + delete_space
            + pynutil.insert(" ")
            + currency_minor
        )

        # Pattern: integer_part fractional_part currency currency_minor
        graph_with_cents_v2 = (
            integer_part
            + delete_space
            + pynutil.insert(" ")
            + fractional_part
            + delete_space
            + pynutil.insert(" ")
            + currency
            + delete_space
            + pynutil.insert(" και ")
            + currency_minor
        )

        # Reorder to put amount before currency
        # Final output: amount currency [και cents λεπτά]
        graph = graph_with_cents | graph_currency_first | graph_integer_first

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
