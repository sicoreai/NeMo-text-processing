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


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimals, e.g.
        decimal { integer_part: "τρία" fractional_part: "ένα τέσσερα" } -> τρία κόμμα ένα τέσσερα

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        # Integer part
        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Fractional part with "κόμμα" (comma) as separator
        fractional_part = (
            delete_space
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.insert(" κόμμα ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Optional quantity
        quantity = (
            delete_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.insert(" ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(quantity, 0, 1)

        graph = integer_part + fractional_part + optional_quantity
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
