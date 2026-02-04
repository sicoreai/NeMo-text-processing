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
from pynini.lib import pynutil, byte

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.el.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic content (URLs, emails) in Greek, e.g.
        "example@gmail.com" -> electronic { username: "example" domain: "gmail τελεία com" }
        "www.google.gr" -> electronic { protocol: "www" domain: "google τελεία gr" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # Symbol replacements
        symbols = pynini.string_map([
            ("@", " παπάκι "),
            (".", " τελεία "),
            ("-", " παύλα "),
            ("_", " κάτω παύλα "),
            ("/", " κάθετος "),
            (":", " άνω κάτω τελεία "),
        ])

        # Common domain extensions spoken forms
        domains = pynini.string_map([
            (".gr", " τελεία γκρ"),
            (".com", " τελεία κομ"),
            (".org", " τελεία οργκ"),
            (".net", " τελεία νετ"),
            (".eu", " τελεία ι ου"),
            (".gov", " τελεία γκοβ"),
            (".edu", " τελεία έντου"),
        ])

        # Protocol prefixes
        protocols = pynini.string_map([
            ("http://", "χτπ άνω κάτω τελεία κάθετος κάθετος "),
            ("https://", "χτπς άνω κάτω τελεία κάθετος κάθετος "),
            ("www.", "ντάμπλιου ντάμπλιου ντάμπλιου τελεία "),
        ])

        # Characters that can appear in usernames/domains
        valid_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.union("-", "_", ".")

        # Simple character passthrough (letters and digits stay as-is)
        passthrough = NEMO_ALPHA | NEMO_DIGIT

        # Build a graph that processes electronic strings
        # For each character: either pass through, or replace symbol
        char_processor = passthrough | symbols

        # Process a string character by character
        string_processor = pynini.closure(char_processor)

        # Clean up multiple spaces
        clean_spaces = pynini.cdrewrite(
            pynini.cross(pynini.closure(" ", 2), " "), "", "", NEMO_SIGMA
        )

        # Email pattern: username@domain
        email_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.union("-", "_", ".")
        username = pynini.closure(email_chars, 1)
        domain = pynini.closure(email_chars, 1)

        email_graph = (
            pynutil.insert("username: \"")
            + (username @ string_processor @ clean_spaces)
            + pynutil.insert("\"")
            + pynutil.delete("@")
            + pynutil.insert(" domain: \"παπάκι ")
            + (domain @ string_processor @ clean_spaces)
            + pynutil.insert("\"")
        )

        # URL pattern: protocol://domain/path
        url_chars = NEMO_NOT_SPACE - pynini.union("<", ">", "\"", "'")
        url = pynini.closure(url_chars, 1)

        url_graph = (
            pynutil.insert("protocol: \"")
            + protocols
            + pynutil.insert("\" domain: \"")
            + ((url - pynini.closure(pynini.union("http://", "https://", "www.") + url_chars)) @ string_processor @ clean_spaces)
            + pynutil.insert("\"")
        )

        # Simple URL without protocol
        simple_url_graph = (
            pynutil.insert("domain: \"")
            + (url @ string_processor @ clean_spaces)
            + pynutil.insert("\"")
        )

        graph = email_graph | url_graph | simple_url_graph

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
