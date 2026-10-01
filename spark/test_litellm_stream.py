"""Exercise the real installed splitter without importing LiteLLM's runtime."""
import ast
import copy
import unittest
from collections import deque
from types import SimpleNamespace as NS
from typing import Any, AsyncIterator, Iterator, List, Optional

from patch_litellm_stream import MODULE

tree = ast.parse(MODULE.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_CombinedChunkSplitter')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODULE), 'exec'), globals())

def chunk(content=None, reasoning=None, tools=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, reasoning_content=reasoning,
                                  tool_calls=tools, thinking_blocks=None), finish_reason=finish)])

class SplitterTest(unittest.TestCase):
    def test_mixed_reasoning_answer_preserves_both_and_original(self):
        original = chunk('GATEWAY_STREAM', '\n')
        parts = _CombinedChunkSplitter._split(original)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].choices[0].delta.reasoning_content, '\n')
        self.assertIsNone(parts[0].choices[0].delta.content)
        self.assertEqual(parts[1].choices[0].delta.content, 'GATEWAY_STREAM')
        self.assertIsNone(parts[1].choices[0].delta.reasoning_content)
        self.assertEqual(original.choices[0].delta.content, 'GATEWAY_STREAM')

    def test_mixed_tool_and_finish_preserves_order(self):
        parts = _CombinedChunkSplitter._split(chunk(reasoning='done', tools=[{'id':'x'}], finish='tool_calls'))
        self.assertEqual(len(parts), 3)
        self.assertEqual([p.choices[0].finish_reason for p in parts], [None, None, 'tool_calls'])
        self.assertEqual(parts[1].choices[0].delta.tool_calls, [{'id':'x'}])
        self.assertIsNone(parts[2].choices[0].delta.tool_calls)

    def test_existing_content_finish_and_single_kind_unchanged(self):
        self.assertEqual(len(_CombinedChunkSplitter._split(chunk('text', finish='stop'))), 2)
        original = chunk(reasoning='only')
        self.assertEqual(_CombinedChunkSplitter._split(original), [original])

    def test_sync_iteration_drains_every_part(self):
        parts = list(_CombinedChunkSplitter(iter([chunk('answer', 'thought'), chunk(finish='stop')])))
        self.assertEqual(len(parts), 3)

    def test_async_iteration_drains_every_part(self):
        import asyncio
        async def upstream():
            yield chunk('answer', 'thought')
            yield chunk(finish='stop')
        async def collect():
            return [p async for p in _CombinedChunkSplitter(upstream())]
        self.assertEqual(len(asyncio.run(collect())), 3)

if __name__ == '__main__':
    unittest.main()
