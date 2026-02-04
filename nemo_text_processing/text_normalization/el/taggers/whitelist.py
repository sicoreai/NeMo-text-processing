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
    NEMO_SIGMA,
    GraphFst,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens in Greek.
    Whitelist entries are read directly from TSV file and passed through.
        e.g. "κ." -> tokens { name: "κύριος" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
        input_file: path to a file with whitelist replacements (optional)
    """

    def __init__(
        self,
        input_case: str = "cased",
        deterministic: bool = True,
        input_file: str = None,
    ):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        # Load default whitelist
        whitelist_path = input_file if input_file else get_abs_path("data/whitelist.tsv")

        try:
            whitelist = pynini.string_file(whitelist_path)
        except Exception:
            # If file doesn't exist or is empty, create empty FST
            whitelist = pynini.accep("")

        # Handle case sensitivity
        if input_case == "lower_cased":
            # Create lowercase version of whitelist
            # Greek lowercase mapping
            upper = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏ"
            lower = "αβγδεζηθικλμνξοπρστυφχψωάέήίόύώ"
            to_lower = pynini.union(*[pynini.cross(u, l) for u, l in zip(upper, lower)])
            to_lower = pynini.cdrewrite(to_lower, "", "", NEMO_SIGMA)
            whitelist = pynini.compose(to_lower, whitelist)

        graph = pynutil.insert("name: \"") + whitelist + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
