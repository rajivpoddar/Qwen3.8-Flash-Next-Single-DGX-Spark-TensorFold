"""Normalize TensorFold's combined reasoning/text chunks before Anthropic SSE."""
from pathlib import Path

MODULE = Path('/app/.venv/lib/python3.13/site-packages/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py')
ANCHOR = '''        if not _CombinedChunkSplitter._is_combined(chunk):
            return [chunk]
'''
REPLACEMENT = '''        # TensorFold MTP can finish reasoning and start the answer in one
        # chunk. The Anthropic adapter handles only one block kind at a time.
        choices = getattr(chunk, "choices", None)
        delta = getattr(choices[0], "delta", None) if choices else None
        if delta is not None and getattr(delta, "reasoning_content", None) and (
            getattr(delta, "content", None) or getattr(delta, "tool_calls", None)
        ):
            reasoning_chunk = copy.deepcopy(chunk)
            reasoning_chunk.choices[0].finish_reason = None
            reasoning_chunk.choices[0].delta.content = None
            reasoning_chunk.choices[0].delta.tool_calls = None
            answer_chunk = copy.deepcopy(chunk)
            answer_chunk.choices[0].delta.reasoning_content = None
            return [reasoning_chunk, *_CombinedChunkSplitter._split(answer_chunk)]
        if not _CombinedChunkSplitter._is_combined(chunk):
            return [chunk]
'''

if __name__ == '__main__':
    source = MODULE.read_text()
    if source.count(ANCHOR) != 1:
        raise SystemExit('pinned LiteLLM splitter changed; refusing to patch')
    MODULE.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
