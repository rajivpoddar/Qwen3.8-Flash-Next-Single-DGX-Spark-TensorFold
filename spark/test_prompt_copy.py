"""CPU-only tests of the actual installed Flash Next copy-draft control flow.

Run with --installed in the image or a TensorFold src/tensorfold directory.
Only CUDA forwards/buffers are stubbed; CopyIndex, Stream and the decoder
methods are extracted unchanged from the runtime source.
"""
import ast
import os
from pathlib import Path
import random
import sys
import time
import types
import unittest
from unittest.mock import patch

if sys.argv[1:] == ["--installed"]:
    import tensorfold
    ROOT = Path(tensorfold.__file__).parent
else:
    ROOT = Path(sys.argv[1])
sys.argv = sys.argv[:1]


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"
                 or isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


copy_ns = extract(ROOT / "families/qwen3_5/cuda/decode.py", {"CopyIndex", "copy_chain"}, {})
CopyIndex, copy_chain = copy_ns["CopyIndex"], copy_ns["copy_chain"]
stream_module = types.ModuleType("prompt_copy_test_streams")
sys.modules[stream_module.__name__] = stream_module
exec(compile((ROOT / "cuda/streams.py").read_text(), str(ROOT / "cuda/streams.py"), "exec"), vars(stream_module))
Stream = stream_module.Stream
fake_copy_module = types.ModuleType("tensorfold.families.qwen3_5.cuda.decode")
fake_copy_module.CopyIndex = CopyIndex


class Logits(list):
    def __getitem__(self, index):
        if isinstance(index, list):
            return Logits(super(Logits, self).__getitem__(i) for i in index)
        return super().__getitem__(index)


class Rows:
    def __getitem__(self, index):
        return list(range(index.start, index.stop))


class State:
    def __init__(self):
        self.pos, self.mtp_len, self.mtp_drafted = 30, 12, 2

    def set_mtp_len(self, n):
        self.mtp_len = n


def mtp_stage(w, buf, windows):
    out, row = [], 0
    for st, tokens, _ in windows:
        out.append((st, row, row + len(tokens)))
        row += len(tokens)
    return out


multi_ns = {"time": time, "os": os, "COPY_MATCH": 8, "CONFIDENCE": .7, "DEPTH": 6,
            "PREFILL_ROWS": 2048, "SHARE": 0, "mtp_stage": mtp_stage, "Stream": Stream}
tree = ast.parse((ROOT / "families/qwen4_exp/cuda/multi.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
flag = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
flag.body = [n for n in flag.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == "copy" for t in n.targets)]
flag.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[])
assert len(flag.body) == 1, "runtime does not consume TENSORFOLD_MTP_COPY"
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"__init__", "admit", "_draft_all"}]
for n in cls.body:
    n.decorator_list = []
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                    type_ignores=[])
exec(compile(ast.fix_missing_locations(module), "runtime-multi.py", "exec"), multi_ns)
Decoder = multi_ns["MultiDecoder"]


def decoder():
    d = Decoder()
    d.depth, d.confidence, d.capacity = 3, .60, 262144
    d.w = types.SimpleNamespace(mtp=object())
    d.mbuf = types.SimpleNamespace(streams=Rows())
    d.buf = types.SimpleNamespace(streams=Rows())
    d.pbuf, d.prefill_rows = None, 2048
    d.calls, d.pick_rows = [], []

    def compute(segs):
        d.calls.append([(st, a, b) for st, a, b in segs])
        return Logits(100 + st.sid for st, _, _ in segs)

    def picks(logits, positions, samplings):
        assert len(logits) == len(positions) == len(samplings)
        d.pick_rows.append(list(logits))
        return [(int(x), .9) for x in logits]

    d._mtp, d._picks = compute, picks
    return d


def stream(sid=0, match=True, count=30):
    prompt = list(range(20)) + [90] + list(range(8))
    if not match:
        prompt = list(range(1000, 1030))
    s = Stream(prompt, count, sid=sid)
    s.st = State()
    s.st.sid = sid
    s.context, s.out = prompt[:-1], [prompt[-1]]
    s.copies, s.copy_pending = CopyIndex(8), False
    return s


class CopyTests(unittest.TestCase):
    def test_flag_is_opt_in_and_only_literal_one(self):
        for value in ("0", "1", "true", ""):
            with patch.dict(os.environ, {"TENSORFOLD_MTP_COPY": value}):
                self.assertEqual(Decoder().copy, value == "1")

    def test_real_index_matches_scan_as_verified_context_grows(self):
        rng, context, index = random.Random(4), list(range(100)), CopyIndex(8)
        for _ in range(300):
            if rng.random() < .5:
                at = rng.randrange(len(context) - 20)
                context.extend(context[at:at + rng.randint(1, 10)])
            else:
                context.extend(rng.randrange(100) for _ in range(rng.randint(1, 10)))
            self.assertEqual(index.propose(context, 8), copy_chain(context, 8))

    def test_copy_absorbs_kept_state_and_skips_later_mtp_calls(self):
        d, s = decoder(), stream()
        before = list(s.context)
        d._draft_all([(s, 0, [7])], {s.sid: [7]})
        self.assertEqual(s.drafts, [8, 9, 10])
        self.assertTrue(s.copy_pending)
        self.assertEqual(s.context, before)
        self.assertEqual((s.st.mtp_len, s.st.mtp_drafted), (11, 0))
        self.assertEqual(len(d.calls), 1)
        self.assertEqual(d.pick_rows, [])

    def test_copy_obeys_remaining_reply_room(self):
        d, s = decoder(), stream(count=4)
        d._draft_all([(s, 0, [7])], {s.sid: [7]})
        self.assertEqual(s.drafts, [8, 9])

    def test_mixed_copy_and_mtp_use_each_fallbacks_own_logits(self):
        for copy_sid in (0, 1, 2, 3):
            d = decoder()
            ss = [stream(i, match=i == copy_sid) for i in range(4)]
            d._draft_all([(s, i, [7]) for i, s in enumerate(ss)], {s.sid: [7] for s in ss})
            expected = [100 + i for i in range(4) if i != copy_sid]
            self.assertEqual(d.pick_rows, [expected] * 3)
            for s in ss:
                self.assertEqual(s.drafts, [8, 9, 10] if s.sid == copy_sid else [100 + s.sid] * 3)

    def test_no_match_keeps_the_mtp_path(self):
        d, s = decoder(), stream(match=False)
        d._draft_all([(s, 0, [7])], {s.sid: [7]})
        self.assertEqual(s.drafts, [100] * 3)
        self.assertFalse(s.copy_pending)
        self.assertEqual((s.st.mtp_len, s.st.mtp_drafted), (13, 2))

    def test_disabled_copy_keeps_the_mtp_path(self):
        d, s = decoder(), stream()
        s.copies = None
        d._draft_all([(s, 0, [7])], {})
        self.assertEqual(s.drafts, [100] * 3)
        self.assertFalse(s.copy_pending)

    def test_no_room_or_no_mtp_makes_no_proposals(self):
        for no_mtp in (False, True):
            d, s = decoder(), stream(count=2 if not no_mtp else 30)
            if no_mtp:
                d.mbuf = None
            d._draft_all([(s, 0, [7])], {s.sid: [7]})
            self.assertEqual(s.drafts, [])
            self.assertEqual(d.calls, [])

    def test_context_is_restored_if_lookup_raises(self):
        d, s = decoder(), stream()
        before = list(s.context)
        s.copies = types.SimpleNamespace(propose=lambda *args: (_ for _ in ()).throw(ValueError("bad index")))
        with self.assertRaisesRegex(ValueError, "bad index"):
            d._draft_all([(s, 0, [7])], {s.sid: [7]})
        self.assertEqual(s.context, before)

    def test_admission_only_indexes_enabled_mtp_requests(self):
        for enabled, drafting, depth in ((True, True, 3), (False, True, 3), (True, False, 3), (True, True, 0)):
            d, s = decoder(), Stream(list(range(50)), 10, draft=drafting)
            d.copy, d.depth = enabled, depth
            d.streams, d.filling, d.fills, d.next_id = {}, [], {}, 0
            d._slot_for = lambda *args: (State(), None, 0)
            d._grow = lambda *args, **kwargs: True
            with patch.dict(multi_ns, {"_slot": lambda *args: None, "prefill_begin": lambda *args, **kwargs: 0}), \
                    patch.dict(sys.modules, {fake_copy_module.__name__: fake_copy_module}):
                d.admit(s)
            self.assertEqual(s.copies is not None, enabled and drafting and depth > 0)
            self.assertFalse(s.copy_pending)

    def test_copy_stats_count_only_verified_rows_and_survive_replay(self):
        s = stream()
        s.copy_drafted, s.copy_accepted = 5, 3
        self.assertEqual((s.stats()["copy_drafted"], s.stats()["copy_accepted"]), (5, 3))
        replay = s.continued()
        replay.copies = CopyIndex(8)
        replay.copy_drafted, replay.copy_accepted = 4, 2
        self.assertEqual((replay.stats()["copy_drafted"], replay.stats()["copy_accepted"]), (9, 5))
        s.copies = None
        self.assertNotIn("copy_drafted", s.stats())

    def test_round_counts_copy_acceptance_not_unverified_proposals(self):
        tree = ast.parse((ROOT / "families/qwen4_exp/cuda/multi.py").read_text())
        counter = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                       and isinstance(n.test, ast.Attribute) and n.test.attr == "copy_pending")
        code = compile(ast.fix_missing_locations(ast.Module(body=[counter], type_ignores=[])), "counter", "exec")
        for pending in (False, True):
            s = stream()
            s.copy_pending = pending
            exec(code, {"s": s, "tokens": [7, 8, 9, 10], "path": [0, 1]})
            self.assertEqual((s.copy_drafted, s.copy_accepted), (3, 1) if pending else (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
