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


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time in Greek, e.g.
        "14:30" -> time { hours: "δεκατέσσερις" minutes: "τριάντα" }
        "9:05" -> time { hours: "εννέα" minutes: "πέντε" }
        "14:30:45" -> time { hours: "δεκατέσσερις" minutes: "τριάντα" seconds: "σαράντα πέντε" }

    Greek typically uses 24-hour format.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph

        # Time separator
        time_separator = pynini.union(":", ".")

        # Delete leading zero for hours/minutes
        delete_leading_zero_single = pynini.cdrewrite(pynutil.delete("0"), "[BOS]", NEMO_DIGIT, NEMO_SIGMA)

        # Hours (0-23)
        hours = pynini.closure(NEMO_DIGIT, 1, 2)
        hours_graph = hours @ delete_leading_zero_single @ cardinal_graph

        # Minutes (00-59)
        minutes = NEMO_DIGIT ** 2
        minutes_graph = minutes @ delete_leading_zero_single @ cardinal_graph

        # Seconds (00-59)
        seconds = NEMO_DIGIT ** 2
        seconds_graph = seconds @ delete_leading_zero_single @ cardinal_graph

        # Hours field
        hours_field = pynutil.insert("hours: \"") + hours_graph + pynutil.insert("\"")

        # Minutes field
        minutes_field = pynutil.insert(" minutes: \"") + minutes_graph + pynutil.insert("\"")

        # Seconds field (optional)
        seconds_field = pynutil.insert(" seconds: \"") + seconds_graph + pynutil.insert("\"")

        # Time patterns
        # HH:MM
        graph_hm = (
            hours_field
            + pynutil.delete(time_separator)
            + minutes_field
        )

        # HH:MM:SS
        graph_hms = (
            hours_field
            + pynutil.delete(time_separator)
            + minutes_field
            + pynutil.delete(time_separator)
            + seconds_field
        )

        # Optional suffix (π.μ., μ.μ.)
        time_suffix = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        suffix_graph = pynini.invert(time_suffix)
        optional_suffix = pynini.closure(
            pynutil.delete(pynini.closure(" ", 0, 1))
            + pynutil.insert(" suffix: \"")
            + suffix_graph
            + pynutil.insert("\""),
            0,
            1,
        )

        graph = (graph_hms | graph_hm) + optional_suffix

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
