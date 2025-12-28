import unittest

from src.data.generate_synthetic import build_doc_adjacency


class TestRagAdjacency(unittest.TestCase):
    def test_page_based_adjacency(self):
        chunks = [
            {"page": 2, "text": "b"},
            {"page": 1, "text": "a"},
            {"page": 3, "text": "c"},
        ]
        prev_by_idx, next_by_idx = build_doc_adjacency(chunks)

        # Document order: idx1(page1) -> idx0(page2) -> idx2(page3)
        self.assertIsNone(prev_by_idx[1])
        self.assertEqual(next_by_idx[1], 0)
        self.assertEqual(prev_by_idx[0], 1)
        self.assertEqual(next_by_idx[0], 2)
        self.assertEqual(prev_by_idx[2], 0)
        self.assertIsNone(next_by_idx[2])

    def test_chunk_based_adjacency_uses_chunk_index(self):
        chunks = [
            {"page_start": 10, "chunk_index": 1, "text": "b"},
            {"page_start": 10, "chunk_index": 0, "text": "a"},
            {"page_start": 11, "chunk_index": 0, "text": "c"},
        ]
        prev_by_idx, next_by_idx = build_doc_adjacency(chunks)

        # Document order: idx1(10,0) -> idx0(10,1) -> idx2(11,0)
        self.assertIsNone(prev_by_idx[1])
        self.assertEqual(next_by_idx[1], 0)
        self.assertEqual(prev_by_idx[0], 1)
        self.assertEqual(next_by_idx[0], 2)


if __name__ == "__main__":
    unittest.main()


