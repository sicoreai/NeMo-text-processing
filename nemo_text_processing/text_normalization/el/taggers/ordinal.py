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
    insert_space,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals in Greek, e.g.
        "1ος" -> ordinal { integer: "πρώτος" morphosyntactic_features: "masc" }
        "1η" -> ordinal { integer: "πρώτη" morphosyntactic_features: "fem" }
        "1ο" -> ordinal { integer: "πρώτο" morphosyntactic_features: "neut" }

    Greek ordinals have three genders: masculine (-ος), feminine (-η), neuter (-ο)

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Ordinal stems (masculine form as base)
        ordinal_roots = pynini.string_map([
            ("1", "πρώτ"),
            ("2", "δεύτερ"),
            ("3", "τρίτ"),
            ("4", "τέταρτ"),
            ("5", "πέμπτ"),
            ("6", "έκτ"),
            ("7", "έβδομ"),
            ("8", "όγδο"),
            ("9", "ένατ"),
            ("10", "δέκατ"),
            ("11", "ενδέκατ"),
            ("12", "δωδέκατ"),
            ("20", "εικοστ"),
            ("30", "τριακοστ"),
            ("40", "τεσσαρακοστ"),
            ("50", "πεντηκοστ"),
            ("60", "εξηκοστ"),
            ("70", "εβδομηκοστ"),
            ("80", "ογδοηκοστ"),
            ("90", "ενενηκοστ"),
            ("100", "εκατοστ"),
        ])

        ordinal_digit_roots = pynini.string_map([
            ("1", "πρώτ"),
            ("2", "δεύτερ"),
            ("3", "τρίτ"),
            ("4", "τέταρτ"),
            ("5", "πέμπτ"),
            ("6", "έκτ"),
            ("7", "έβδομ"),
            ("8", "όγδο"),
            ("9", "ένατ"),
        ])

        ordinal_tens_roots = pynini.string_map([
            ("1", "δέκατ"),
            ("2", "εικοστ"),
            ("3", "τριακοστ"),
            ("4", "τεσσαρακοστ"),
            ("5", "πεντηκοστ"),
            ("6", "εξηκοστ"),
            ("7", "εβδομηκοστ"),
            ("8", "ογδοηκοστ"),
            ("9", "ενενηκοστ"),
        ])

        # Gender endings
        masc_ending = pynutil.insert("ος")
        fem_ending = pynutil.insert("η")
        neut_ending = pynutil.insert("ο")

        # Single digit ordinals (1-9)
        graph_digit_masc = ordinal_digit_roots + masc_ending
        graph_digit_fem = ordinal_digit_roots + fem_ending
        graph_digit_neut = ordinal_digit_roots + neut_ending

        # Teens (11-19) - special forms
        teens_roots = pynini.string_map([
            ("11", "ενδέκατ"),
            ("12", "δωδέκατ"),
            ("13", "δέκατος τρίτ"),  # compound
            ("14", "δέκατος τέταρτ"),
            ("15", "δέκατος πέμπτ"),
            ("16", "δέκατος έκτ"),
            ("17", "δέκατος έβδομ"),
            ("18", "δέκατος όγδο"),
            ("19", "δέκατος ένατ"),
        ])

        # For compound teens 13-19, we need special handling
        simple_teens = pynini.string_map([
            ("11", "ενδέκατ"),
            ("12", "δωδέκατ"),
        ])

        # Decades as ordinals (10, 20, 30, etc.)
        graph_decades_masc = pynini.cross("0", "") + ordinal_tens_roots + masc_ending
        graph_decades_fem = pynini.cross("0", "") + ordinal_tens_roots + fem_ending
        graph_decades_neut = pynini.cross("0", "") + ordinal_tens_roots + neut_ending

        # Two digit ordinals (21-99, excluding pure decades)
        # Pattern: decade-ordinal + digit-ordinal (e.g., εικοστός πρώτος)
        graph_compound_tens_masc = (
            ordinal_tens_roots + masc_ending + insert_space + ordinal_digit_roots + masc_ending
        )
        graph_compound_tens_fem = (
            ordinal_tens_roots + fem_ending + insert_space + ordinal_digit_roots + fem_ending
        )
        graph_compound_tens_neut = (
            ordinal_tens_roots + neut_ending + insert_space + ordinal_digit_roots + neut_ending
        )

        # Build two-digit graphs
        two_digit_not_teen = (NEMO_DIGIT - "0" - "1") + (NEMO_DIGIT - "0")
        two_digit_teen = pynini.accep("1") + NEMO_DIGIT

        graph_teens_masc = simple_teens + masc_ending
        graph_teens_fem = simple_teens + fem_ending
        graph_teens_neut = simple_teens + neut_ending

        # For 13-19, use compound form
        teens_13_19_masc = pynini.string_map([
            ("13", "δέκατος τρίτος"),
            ("14", "δέκατος τέταρτος"),
            ("15", "δέκατος πέμπτος"),
            ("16", "δέκατος έκτος"),
            ("17", "δέκατος έβδομος"),
            ("18", "δέκατος όγδοος"),
            ("19", "δέκατος ένατος"),
        ])
        teens_13_19_fem = pynini.string_map([
            ("13", "δέκατη τρίτη"),
            ("14", "δέκατη τέταρτη"),
            ("15", "δέκατη πέμπτη"),
            ("16", "δέκατη έκτη"),
            ("17", "δέκατη έβδομη"),
            ("18", "δέκατη όγδοη"),
            ("19", "δέκατη ένατη"),
        ])
        teens_13_19_neut = pynini.string_map([
            ("13", "δέκατο τρίτο"),
            ("14", "δέκατο τέταρτο"),
            ("15", "δέκατο πέμπτο"),
            ("16", "δέκατο έκτο"),
            ("17", "δέκατο έβδομο"),
            ("18", "δέκατο όγδοο"),
            ("19", "δέκατο ένατο"),
        ])

        graph_all_teens_masc = graph_teens_masc | teens_13_19_masc | pynini.cross("10", "δέκατος")
        graph_all_teens_fem = graph_teens_fem | teens_13_19_fem | pynini.cross("10", "δέκατη")
        graph_all_teens_neut = graph_teens_neut | teens_13_19_neut | pynini.cross("10", "δέκατο")

        # Build complete ordinal graphs by gender
        graph_ordinal_masc = pynini.union(
            graph_digit_masc,
            graph_all_teens_masc,
            graph_decades_masc,
            two_digit_not_teen @ graph_compound_tens_masc,
        )

        graph_ordinal_fem = pynini.union(
            graph_digit_fem,
            graph_all_teens_fem,
            graph_decades_fem,
            two_digit_not_teen @ graph_compound_tens_fem,
        )

        graph_ordinal_neut = pynini.union(
            graph_digit_neut,
            graph_all_teens_neut,
            graph_decades_neut,
            two_digit_not_teen @ graph_compound_tens_neut,
        )

        self.graph_masc = graph_ordinal_masc.optimize()
        self.graph_fem = graph_ordinal_fem.optimize()
        self.graph_neut = graph_ordinal_neut.optimize()

        # Detect gender from suffix (ος, η, ο after number)
        delete_ordinal_indicator = pynutil.delete(pynini.union(".", "ος", "ης", "ο", "η", "α", "°"))

        # Build final graph with morphosyntactic features
        # Format: number + suffix -> ordinal { integer: "word" morphosyntactic_features: "gender" }

        graph_masc_tagged = (
            pynutil.insert("integer: \"")
            + graph_ordinal_masc
            + pynutil.insert("\" morphosyntactic_features: \"masc\"")
        )

        graph_fem_tagged = (
            pynutil.insert("integer: \"")
            + graph_ordinal_fem
            + pynutil.insert("\" morphosyntactic_features: \"fem\"")
        )

        graph_neut_tagged = (
            pynutil.insert("integer: \"")
            + graph_ordinal_neut
            + pynutil.insert("\" morphosyntactic_features: \"neut\"")
        )

        # Input patterns: 1ος, 1η, 1ο, 1., 1ο, etc.
        number = pynini.closure(NEMO_DIGIT, 1, 2)

        # Masculine markers
        masc_suffix = pynini.union("ος", "ός")
        fem_suffix = pynini.union("η", "ή", "α", "ά")
        neut_suffix = pynini.union("ο", "ό")

        graph = pynini.union(
            (number + pynutil.delete(masc_suffix)) @ graph_masc_tagged,
            (number + pynutil.delete(fem_suffix)) @ graph_fem_tagged,
            (number + pynutil.delete(neut_suffix)) @ graph_neut_tagged,
            # Period as ordinal marker (defaults to masculine)
            (number + pynutil.delete(".")) @ graph_masc_tagged,
        )

        if not deterministic:
            # In non-deterministic mode, also allow unmarked numbers to produce ordinals
            graph |= (number + pynutil.delete(pynini.closure(pynini.union("ος", "η", "ο", "."), 0, 1))) @ (
                graph_masc_tagged | graph_fem_tagged | graph_neut_tagged
            )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
