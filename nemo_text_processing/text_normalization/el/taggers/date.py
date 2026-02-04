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
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying dates in Greek, e.g.
        "15/3/2024" -> date { day: "δεκαπέντε" month: "Μαρτίου" year: "δύο χιλιάδες είκοσι τέσσερα" }
        "15-03-2024" -> date { day: "δεκαπέντε" month: "Μαρτίου" year: "δύο χιλιάδες είκοσι τέσσερα" }

    Greek dates use DD/MM/YYYY format. Days are cardinal numbers, months are in genitive case.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph

        # Month names in genitive case
        months = pynini.string_file(get_abs_path("data/dates/months.tsv"))
        month_graph = pynini.invert(months)

        # Date separators
        separator = pynini.union("/", "-", ".")

        # Day (1-31)
        delete_leading_zero = pynini.cdrewrite(pynutil.delete("0"), "[BOS]", "", NEMO_DIGIT)
        day = pynini.closure(NEMO_DIGIT, 1, 2) @ delete_leading_zero @ cardinal_graph

        # Month (1-12) -> month name in genitive
        month_num = pynini.closure(NEMO_DIGIT, 1, 2)
        month = month_num @ delete_leading_zero @ month_graph

        # Year - can be 2 or 4 digits
        year_2digit = NEMO_DIGIT ** 2
        year_4digit = NEMO_DIGIT ** 4
        year = (year_2digit | year_4digit) @ cardinal_graph

        # Day field
        day_field = pynutil.insert("day: \"") + day + pynutil.insert("\"")

        # Month field
        month_field = pynutil.insert(" month: \"") + month + pynutil.insert("\"")

        # Year field
        year_field = pynutil.insert(" year: \"") + year + pynutil.insert("\"")

        # Full date patterns
        # DD/MM/YYYY
        graph_dmy = (
            day_field
            + pynutil.delete(separator)
            + month_field
            + pynutil.delete(separator)
            + year_field
        )

        # DD/MM (without year)
        graph_dm = (
            day_field
            + pynutil.delete(separator)
            + month_field
        )

        # DD month_name YYYY (e.g., "15 Μαρτίου 2024")
        month_name = pynini.string_file(get_abs_path("data/dates/months.tsv"))
        month_name_graph = pynini.project(month_name, "input")

        graph_text_date = (
            day_field
            + pynutil.delete(pynini.closure(" ", 1))
            + pynutil.insert(" month: \"")
            + month_name_graph
            + pynutil.insert("\"")
            + pynini.closure(
                pynutil.delete(pynini.closure(" ", 1))
                + year_field,
                0,
                1,
            )
        )

        graph = graph_dmy | graph_dm | graph_text_date

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
