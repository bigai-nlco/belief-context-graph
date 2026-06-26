from __future__ import annotations

import unittest

from bcg import BCG, BCGMemory


class PublicInterfaceTests(unittest.TestCase):
    def test_public_imports(self) -> None:
        self.assertIsNotNone(BCG)
        self.assertIsNotNone(BCGMemory)

    def test_bcg_initializes_graph(self) -> None:
        bcg = BCG()

        self.assertEqual(bcg.nodes, [])
        self.assertEqual(bcg.edges, [])

    def test_memory_can_hold_graph(self) -> None:
        memory = BCGMemory(graph=BCG())

        self.assertIsInstance(memory.graph, BCG)

    def test_memory_believe_returns_no_matches_for_empty_graph(self) -> None:
        memory = BCGMemory()

        self.assertEqual(memory.believe("truth(example_claim)"), [])


if __name__ == "__main__":
    unittest.main()
