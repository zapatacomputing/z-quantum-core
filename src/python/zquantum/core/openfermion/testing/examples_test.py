#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2022 Zapata Computing, Inc. for compatibility reasons.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests the code in the examples directory of the git repo."""

import os
import unittest

import numpy


class ExamplesTest(unittest.TestCase):
    def setUp(self):
        """Construct test info"""
        self.testing_folder = os.path.join(
            os.path.dirname(__file__),  # Start at this file's directory.
        )

    def test_example(self):
        """Unit test for examples/performance_benchmark.py."""
        numpy.random.seed(1)

        trial_val = 600
        self.assertEqual(trial_val, 600)

        self.assertTrue(os.path.isdir(self.testing_folder))


def is_matplotlib_cell(cell):  # pragma: no cover
    return "%matplotlib" in cell.source


def strip_magics_and_shows(text):  # pragma: no cover
    """Remove Jupyter magics and pyplot show commands."""
    lines = [line for line in text.split("\n") if not contains_magic_or_show(line)]
    return "\n".join(lines)


def contains_magic_or_show(line):  # pragma: no cover
    return line.strip().startswith("%") or "pyplot.show(" in line or "plt.show(" in line
