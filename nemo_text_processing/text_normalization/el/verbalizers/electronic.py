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
)


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic content, e.g.
        electronic { username: "example" domain: "gmail τελεία com" }
        -> example παπάκι gmail τελεία com

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # Username
        username = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Domain
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Protocol
        protocol = (
            pynutil.delete("protocol:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Email: username domain
        graph_email = username + delete_space + pynutil.insert(" ") + domain

        # URL with protocol: protocol domain
        graph_url_with_protocol = protocol + delete_space + pynutil.insert(" ") + domain

        # URL without protocol: just domain
        graph_url_simple = domain

        graph = graph_email | graph_url_with_protocol | graph_url_simple

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
