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
from pynini.lib import byte

from nemo_text_processing.text_normalization.en.graph_utils import delete_space, insert_space

# Greek alphabet - lowercase (including final sigma ς)
_EL_ALPHA_LOWER = "αβγδεζηθικλμνξοπρσςτυφχψω"
# Greek alphabet - uppercase
_EL_ALPHA_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
# Greek accented vowels - lowercase
_EL_ACCENTED_LOWER = "άέήίόύώϊϋΐΰ"
# Greek accented vowels - uppercase
_EL_ACCENTED_UPPER = "ΆΈΉΊΌΎΏΪΫ"
# All Greek vowels (for reference)
_EL_VOWELS = "αεηιουωάέήίόύώϊϋΐΰΑΕΗΙΟΥΩΆΈΉΊΌΎΏΪΫ"

# Case conversion mappings
_LOWER_CHARS = _EL_ALPHA_LOWER.replace("ς", "") + _EL_ACCENTED_LOWER
_UPPER_CHARS = _EL_ALPHA_UPPER + _EL_ACCENTED_UPPER

TO_LOWER = pynini.union(*[pynini.cross(u, l) for u, l in zip(_UPPER_CHARS, _LOWER_CHARS)])
TO_UPPER = pynini.invert(TO_LOWER)

# Character class FSTs
EL_LOWER = pynini.union(*_EL_ALPHA_LOWER, *_EL_ACCENTED_LOWER).optimize()
EL_UPPER = pynini.union(*_EL_ALPHA_UPPER, *_EL_ACCENTED_UPPER).optimize()
EL_ALPHA = pynini.union(EL_LOWER, EL_UPPER).optimize()
EL_ALNUM = pynini.union(byte.DIGIT, EL_ALPHA).optimize()
EL_VOWELS = pynini.union(*_EL_VOWELS).optimize()

# Utility FSTs
ensure_space = pynini.closure(delete_space, 0, 1) + insert_space

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")
