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


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "τρία" denominator: "τέσσερα" } -> τρία προς τέσσερα
        fraction { integer_part: "δύο" numerator: "ένα" denominator: "δύο" } -> δύο και ένα προς δύο

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Integer part (optional)
        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Numerator
        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Denominator with "προς" (to/over) connector
        denominator = (
            delete_space
            + pynutil.delete("denominator:")
            + delete_space
            + pynutil.insert(" προς ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Empty denominator (for special fractions already resolved)
        empty_denominator = (
            delete_space
            + pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete("\"")
            + pynutil.delete("\"")
        )

        # Simple fraction: numerator προς denominator
        graph_simple = numerator + denominator

        # Special fraction (pre-resolved): just numerator
        graph_special = numerator + empty_denominator

        # Mixed fraction: integer και numerator προς denominator
        graph_mixed = (
            integer_part
            + pynutil.insert(" και ")
            + delete_space
            + numerator
            + denominator
        )

        graph = graph_mixed | graph_simple | graph_special

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
