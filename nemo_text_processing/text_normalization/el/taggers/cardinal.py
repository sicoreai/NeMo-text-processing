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
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.el.graph_utils import EL_ALPHA
from nemo_text_processing.text_normalization.el.utils import get_abs_path


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings
    (groups of three digits delineated by period or space) to a string of digits.

    Args:
        fst: Any pynini.FstLike object

    Returns:
        fst: A pynini.FstLike object
    """
    cardinal_separator = pynini.string_map([".", NEMO_SPACE])
    exactly_three_digits = NEMO_DIGIT ** 3
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)
    up_to_three_digits = up_to_three_digits - "000" - "00" - "0"

    cardinal_string = pynini.closure(NEMO_DIGIT, 1)

    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three_digits + pynutil.delete(cardinal_separator))
        + exactly_three_digits
    )

    return cardinal_string @ fst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in Greek, e.g.
        "123" -> cardinal { integer: "εκατόν είκοσι τρία" }
        "1000" -> cardinal { integer: "χίλια" }
        "-5" -> cardinal { negative: "true" integer: "πέντε" }

    Greek numbers 1-4 and hundreds (200-900) have gender forms.
    Default output uses neuter gender.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Load number data
        zero = pynini.invert(pynini.string_file(get_abs_path("data/number/zero.tsv")))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv")))
        tens = pynini.invert(pynini.string_file(get_abs_path("data/number/tens.tsv")))
        hundreds = pynini.invert(pynini.string_file(get_abs_path("data/number/hundreds.tsv")))

        # Gender-specific forms
        digit_masc = pynini.invert(pynini.string_file(get_abs_path("data/number/gender/digit_masc.tsv")))
        digit_fem = pynini.invert(pynini.string_file(get_abs_path("data/number/gender/digit_fem.tsv")))
        digit_neut = pynini.invert(pynini.string_file(get_abs_path("data/number/gender/digit_neut.tsv")))

        self.digit = digit
        self.digit_masc = digit_masc
        self.digit_fem = digit_fem
        self.digit_neut = digit_neut

        graph_zero = zero
        graph_digit = digit

        # Teens (11-19) from tens file - these are specifically 11-19
        teens = pynini.string_map([
            ("11", "έντεκα"),
            ("12", "δώδεκα"),
            ("13", "δεκατρία"),
            ("14", "δεκατέσσερα"),
            ("15", "δεκαπέντε"),
            ("16", "δεκαέξι"),
            ("17", "δεκαεπτά"),
            ("18", "δεκαοκτώ"),
            ("19", "δεκαεννέα"),
        ])

        # Decade words (10, 20, 30, etc.)
        decades = pynini.string_map([
            ("10", "δέκα"),
            ("20", "είκοσι"),
            ("30", "τριάντα"),
            ("40", "σαράντα"),
            ("50", "πενήντα"),
            ("60", "εξήντα"),
            ("70", "εβδομήντα"),
            ("80", "ογδόντα"),
            ("90", "ενενήντα"),
        ])

        # Two-digit numbers (10-99)
        # Pattern: decade + (optional space + digit)
        graph_tens = pynini.union(
            teens,
            decades,
            decades + insert_space + digit,
        )

        self.tens = graph_tens.optimize()

        # Two digit non-zero
        self.two_digit_non_zero = pynini.union(
            graph_digit,
            graph_tens,
            (pynutil.delete("0") + digit),
        ).optimize()

        # Hundreds (100-999)
        # 100 = εκατό (or εκατόν before vowel - we simplify to εκατόν)
        # 200-900 have gender forms, default neuter

        graph_hundred = pynini.cross("1", "εκατόν")

        # For 200-900, we use the neuter form as default
        graph_hundreds_prefix = pynini.string_map([
            ("2", "διακόσια"),
            ("3", "τριακόσια"),
            ("4", "τετρακόσια"),
            ("5", "πεντακόσια"),
            ("6", "εξακόσια"),
            ("7", "επτακόσια"),
            ("8", "οκτακόσια"),
            ("9", "εννιακόσια"),
        ])

        # 100 alone
        graph_hundred_alone = pynini.cross("100", "εκατό")

        # Hundreds with tens or digits
        graph_hundreds = pynini.union(
            graph_hundred_alone,
            graph_hundred + insert_space + graph_tens,
            graph_hundred + insert_space + digit,
            graph_hundreds_prefix + pynini.cross("00", ""),
            graph_hundreds_prefix + insert_space + graph_tens,
            graph_hundreds_prefix + insert_space + digit,
        )

        self.hundreds = graph_hundreds.optimize()

        # Build component for hundreds with leading zeros handled
        graph_hundreds_component = pynini.union(
            graph_hundreds,
            pynutil.delete("0") + graph_tens,
        )

        graph_hundreds_component_at_least_one_non_zero = pynini.union(
            graph_hundreds_component,
            pynutil.delete("00") + graph_digit,
        )

        self.graph_hundreds_component_at_least_one_non_zero = graph_hundreds_component_at_least_one_non_zero.optimize()

        # Thousands
        # 1000 = χίλια
        # 2000-999000 = number + χιλιάδες
        graph_thousand_alone = pynini.cross("001", "χίλια")

        # For 2-999 thousands: number + χιλιάδες
        graph_thousands_prefix = (
            (self.graph_hundreds_component_at_least_one_non_zero - pynini.accep("ένα"))
            + insert_space
            + pynutil.insert("χιλιάδες")
        )

        # Special case for "one thousand" - χίλια (not ένα χιλιάδες)
        graph_thousands = pynini.union(
            graph_thousand_alone,
            graph_thousands_prefix,
        )

        graph_thousands_component = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero,
            graph_thousands + insert_space + (graph_hundreds_component_at_least_one_non_zero | pynutil.delete("000")),
        )

        self.graph_thousands_component = graph_thousands_component.optimize()

        # Millions
        # 1 million = ένα εκατομμύριο
        # 2+ million = number + εκατομμύρια
        graph_million_one = pynini.cross("001", "ένα εκατομμύριο")
        graph_millions_prefix = (
            (self.graph_hundreds_component_at_least_one_non_zero - pynini.accep("ένα"))
            + insert_space
            + pynutil.insert("εκατομμύρια")
        )

        graph_millions = pynini.union(
            pynutil.delete("000") + graph_thousands_component,
            graph_million_one + insert_space + (graph_thousands_component | pynutil.delete("000000")),
            graph_millions_prefix + insert_space + (graph_thousands_component | pynutil.delete("000000")),
        )

        # Billions
        # 1 billion = ένα δισεκατομμύριο
        # 2+ billion = number + δισεκατομμύρια
        graph_billion_one = pynini.cross("001", "ένα δισεκατομμύριο")
        graph_billions_prefix = (
            (self.graph_hundreds_component_at_least_one_non_zero - pynini.accep("ένα"))
            + insert_space
            + pynutil.insert("δισεκατομμύρια")
        )

        graph_billions = pynini.union(
            pynutil.delete("000") + graph_millions,
            graph_billion_one + insert_space + (graph_millions | pynutil.delete("000000000")),
            graph_billions_prefix + insert_space + (graph_millions | pynutil.delete("000000000")),
        )

        # Trillions
        graph_trillion_one = pynini.cross("001", "ένα τρισεκατομμύριο")
        graph_trillions_prefix = (
            (self.graph_hundreds_component_at_least_one_non_zero - pynini.accep("ένα"))
            + insert_space
            + pynutil.insert("τρισεκατομμύρια")
        )

        graph_trillions = pynini.union(
            pynutil.delete("000") + graph_billions,
            graph_trillion_one + insert_space + (graph_billions | pynutil.delete("000000000000")),
            graph_trillions_prefix + insert_space + (graph_billions | pynutil.delete("000000000000")),
        )

        # Full graph - pad to fixed width and convert
        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 15
            @ graph_trillions
        )

        # Clean up extra spaces
        delete_extra_spaces = pynini.cdrewrite(
            pynini.cross(pynini.closure(" ", 2), " "), "", "", NEMO_SIGMA
        )
        clean_leading_space = pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
        clean_trailing_space = pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)

        self.graph = (self.graph @ delete_extra_spaces @ clean_leading_space @ clean_trailing_space).optimize()

        # Add zero
        self.graph |= graph_zero

        # Store unfiltered graph
        self.graph_unfiltered = self.graph

        # Apply punctuation filtering (allows "1.000" format)
        self.graph = filter_punctuation(self.graph).optimize()

        # Build final graph with optional negative sign
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )

        final_graph = (
            optional_minus_graph
            + pynutil.insert("integer: \"")
            + self.graph
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
