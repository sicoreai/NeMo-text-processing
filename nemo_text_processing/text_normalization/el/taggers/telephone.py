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


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers in Greek, e.g.
        "+30 210 1234567" -> telephone { country_code: "τριάντα" number_part: "δύο ένα μηδέν ένα δύο τρία τέσσερα πέντε έξι επτά" }
        "6912345678" -> telephone { number_part: "έξι εννέα ένα δύο τρία τέσσερα πέντε έξι επτά οκτώ" }

    Greek telephone numbers:
    - Country code: +30
    - Mobile: 69xxxxxxxx (10 digits)
    - Landline: 2x xxxxxxxx (10 digits, starting with 2)

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        # Digit to word mapping
        digit_to_word = pynini.string_map([
            ("0", "μηδέν"),
            ("1", "ένα"),
            ("2", "δύο"),
            ("3", "τρία"),
            ("4", "τέσσερα"),
            ("5", "πέντε"),
            ("6", "έξι"),
            ("7", "επτά"),
            ("8", "οκτώ"),
            ("9", "εννέα"),
        ])

        # Read digits one by one with space between
        digit_sequence = pynini.closure(digit_to_word + insert_space) + digit_to_word

        # Delete separators (space, dash, parentheses)
        delete_separator = pynutil.delete(pynini.union(" ", "-", "(", ")", "."))
        optional_separator = pynini.closure(delete_separator, 0, 1)

        # Country code: +30 or 0030
        country_code_30 = pynini.cross("+30", "τριάντα") | pynini.cross("0030", "τριάντα")

        # Phone number digits (read individually)
        phone_digits = pynini.closure(
            (NEMO_DIGIT @ digit_to_word) + optional_separator + insert_space
        ) + (NEMO_DIGIT @ digit_to_word)

        # Clean up extra spaces
        clean_spaces = pynini.cdrewrite(
            pynini.cross(pynini.closure(" ", 2), " "), "", "", NEMO_SIGMA
        )

        # Pattern 1: +30 followed by number
        graph_with_country_code = (
            pynutil.insert("country_code: \"")
            + country_code_30
            + pynutil.insert("\"")
            + optional_separator
            + pynutil.insert(" number_part: \"")
            + (phone_digits @ clean_spaces)
            + pynutil.insert("\"")
        )

        # Pattern 2: Just the number (10 digits for Greek numbers)
        # Greek mobile: starts with 69
        # Greek landline: starts with 2
        greek_prefix = pynini.union(
            pynini.accep("69"),
            pynini.accep("2"),
        )

        graph_number_only = (
            pynutil.insert("number_part: \"")
            + (phone_digits @ clean_spaces)
            + pynutil.insert("\"")
        )

        graph = graph_with_country_code | graph_number_only

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
