"""CPU-only tests of the installed scheduler and real chunk-loop source.

Usage: python test_prefill_yield.py SOURCE_DIRECTORY (scheduler.py, streams.py,
decode.py, multi.py), or --installed inside the patched TensorFold image.
"""
import ast
import os
import pathlib
import queue
import sys
import types
import unittest
from unittest import mock

if sys.argv[1:] == ["--installed"]:
    import tensorfold
    root = pathlib.Path(tensorfold.__file__).parent
    paths = {"scheduler": root / "cuda/scheduler.py", "streams": root / "cuda/streams.py",
             "decode": root / "families/qwen4_exp/cuda/decode.py",
             "multi": root / "families/qwen4_exp/cuda/multi.py"}
else:
    root = pathlib.Path(sys.argv[1])
    paths = {name: root / (name + ".py") for name in ("scheduler", "streams", "decode", "multi")}
sys.argv = sys.argv[:1]

stream_ns = {}
exec(compile(paths["streams"].read_text(), str(paths["streams"]), "exec"), stream_ns)
Stream = stream_ns["Stream"]
tree = ast.parse(paths["scheduler"].read_text())
tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.module != "streams"]
scheduler_ns = {"Stream": Stream}
exec(compile(tree, str(paths["scheduler"]), "exec"), scheduler_ns)
Scheduler = scheduler_ns["Scheduler"]


class FakeDecoder:
    def __init__(self, limit=4):
        self.streams = []
        self.events = []
        self.limit = limit
        self.pending = False
        self.fail_round = False
        self.fail_prefill = False

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        self.events.append("admit")
        self.streams.append(stream)
        stream.take([1])

    def admit_cooperatively(self, stream, on_chunk):
        assert not self.pending, "recursive admission"
        self.pending = True
        assert self.live() + 1 <= self.limit
        try:
            for _ in range(3):
                self.events.append("chunk")
                on_chunk(0.08)
                if self.fail_prefill:
                    raise ValueError("prefill failed")
            self.admit(stream)
        finally:
            self.pending = False

    def round(self):
        self.events.append("decode")
        if self.fail_round:
            raise RuntimeError("decode failed")
        for s in self.streams:
            if not s.done:
                s.take([2])
        return [s for s in self.streams if s.done]

    def finish(self, done):
        for s in done:
            if s in self.streams:
                self.streams.remove(s)

    def drop(self):
        old, self.streams = self.streams, []
        return old


def scheduler(decoder):
    s = object.__new__(Scheduler)  # deterministic: no immortal worker thread in these tests
    s.decoder, s.max_streams = decoder, decoder.limit
    s.decode_share = 0.0  # prior one-round mode; budget-specific cases opt in below
    s.waiting, s.boxes = queue.Queue(), {}
    return s


class SchedulerTests(unittest.TestCase):
    def budget_run(self, chunk_s, round_s, *, count=100, share=0.20):
        d = FakeDecoder()
        stream = Stream([10], count)
        d.admit(stream)
        s = scheduler(d)
        s.decode_share = share
        box = s.boxes[id(stream)] = queue.Queue()
        clock = [0.0]
        original = d.round

        def timed_round():
            clock[0] += round_s
            return original()

        d.round = timed_round
        with mock.patch.object(scheduler_ns["time"], "monotonic", lambda: clock[0]):
            s._decode_between_chunks(chunk_s)
        return d, s, stream, box, clock[0]

    def test_budget_advances_multiple_rounds_and_leaves_queue_untouched(self):
        d, s, stream, box, elapsed = self.budget_run(0.5, 0.03125)
        self.assertEqual(d.events.count("decode"), 4)
        self.assertEqual(len(stream.out), 5)
        self.assertAlmostEqual(elapsed, 0.125)
        self.assertTrue(box.empty())
        self.assertEqual(d.events.count("admit"), 1)

    def test_budget_caps_time_and_stops_when_decode_finishes(self):
        d, _, _, _, elapsed = self.budget_run(100, 0.125)
        self.assertEqual(d.events.count("decode"), 4)
        self.assertEqual(elapsed, 0.5)
        d, s, _, box, _ = self.budget_run(0.4, 0.025, count=3)
        self.assertEqual(d.events.count("decode"), 2)
        self.assertEqual(d.live(), 0)
        self.assertEqual(box.get_nowait()[0], "done")
        self.assertEqual(s.boxes, {})

    def test_budget_keeps_one_round_minimum_and_iteration_bound(self):
        for chunk_s, round_s, share in ((0.4, 0.025, 0), (0, 0.025, .2), (.01, .1, .2)):
            d, _, _, _, _ = self.budget_run(chunk_s, round_s, share=share)
            self.assertEqual(d.events.count("decode"), 1)
        d, _, _, _, _ = self.budget_run(1, 0)
        self.assertEqual(d.events.count("decode"), 32)

    def test_no_live_decode_returns_without_borrowing_buffers(self):
        d = FakeDecoder()
        s = scheduler(d)
        s._decode_between_chunks(100)
        self.assertEqual(d.events, [])

    def test_share_config_rejects_invalid_nonfinite_or_unbounded_values(self):
        for value in ("-0.1", "0.51", "nan", "inf", "oops"):
            with mock.patch.dict(os.environ, {"TENSORFOLD_PREFILL_DECODE_SHARE": value}):
                with self.assertRaises(ValueError):
                    scheduler_ns["_decode_share"]()
        for value in ("0", "0.20", "0.5"):
            with mock.patch.dict(os.environ, {"TENSORFOLD_PREFILL_DECODE_SHARE": value}):
                self.assertEqual(scheduler_ns["_decode_share"](), float(value))

    def test_decode_advances_after_each_chunk_before_first_new_token(self):
        d = FakeDecoder()
        old = Stream([10], 5)
        d.admit(old)
        s = scheduler(d)
        old_box = s.boxes[id(old)] = queue.Queue()
        new = Stream([20], 5)
        s.waiting.put((new, queue.Queue()))
        s._admit()
        self.assertEqual(len(old.out), 4)
        self.assertEqual(d.events[1:], ["chunk", "decode"] * 3 + ["admit"])
        self.assertTrue(old_box.empty())

    def test_finished_or_cancelled_decode_replied_once_during_prefill(self):
        for cancel in (False, True):
            d = FakeDecoder()
            old = Stream([10], 2 if not cancel else 20)
            d.admit(old)
            if cancel:
                old.emit = lambda new: True
            s = scheduler(d)
            box = s.boxes[id(old)] = queue.Queue()
            s.waiting.put((Stream([20], 5), queue.Queue()))
            s._admit()
            self.assertEqual(box.get_nowait()[0], "done")
            self.assertTrue(box.empty())
            self.assertNotIn(old, d.streams)

    def test_decode_failure_fails_existing_and_partial_admission(self):
        d = FakeDecoder()
        old = Stream([10], 5)
        d.admit(old)
        s = scheduler(d)
        box = s.boxes[id(old)] = queue.Queue()
        new_box = queue.Queue()
        s.waiting.put((Stream([20], 5), new_box))
        d.fail_round = True
        s._admit()
        self.assertEqual(box.get_nowait()[0], "error")
        self.assertEqual(new_box.get_nowait()[0], "error")
        self.assertEqual(d.live(), 0)
        self.assertFalse(d.pending)
        self.assertEqual(s.boxes, {})

    def test_prefill_failure_does_not_drop_existing_decode(self):
        d = FakeDecoder()
        old = Stream([10], 5)
        d.admit(old)
        s = scheduler(d)
        s.boxes[id(old)] = queue.Queue()
        box = queue.Queue()
        s.waiting.put((Stream([20], 5), box))
        d.fail_prefill = True
        s._admit()
        self.assertEqual(box.get_nowait()[0], "error")
        self.assertIn(old, d.streams)
        self.assertEqual(len(old.out), 2)
        self.assertFalse(d.pending)

    def test_four_stream_capacity_and_queue_preserved(self):
        d = FakeDecoder()
        s = scheduler(d)
        for i in range(5):
            s.waiting.put((Stream([i], 100), queue.Queue()))
        s._admit()
        self.assertEqual(d.live(), 4)
        self.assertEqual(s.waiting.qsize(), 1)

    def test_decoders_without_hook_keep_original_admission(self):
        d = FakeDecoder()
        d.admit_cooperatively = None
        s = scheduler(d)
        s.waiting.put((Stream([20], 1), queue.Queue()))
        done = s._admit()
        self.assertEqual(d.events, ["admit"])
        self.assertEqual(len(done), 1)
        s._finish(done)
        self.assertEqual(s.boxes, {})


class Tensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return Tensor(self.value)

    def clone(self):
        return Tensor(self.value)


class ChunkLoopTests(unittest.TestCase):
    def run_loop(self, prompt, begin=0, use_mtp=True, callback=True, checkpoint=False):
        events = []
        st = types.SimpleNamespace(pos=begin, mtp_len=begin,
                                   set_mtp_len=lambda n: events.append(("mtp_len", n)))
        pb = types.SimpleNamespace(streams=Tensor("prefill"))
        e = types.SimpleNamespace(w=object(), st=st, pbuf=pb, prefill_rows=2,
                                  sample=lambda logits, positions, sampling: [77])

        def commit(w, state, buf, rows, kept):
            state.pos += rows
            events.append(("commit", state.pos))

        def on_chunk(prefill_s):
            self.assertGreaterEqual(prefill_s, 0)
            self.assertEqual(events[-1], ("sync", st.pos))
            events.append(("decode", st.pos))

        def on_prefix(length, tail):
            self.assertEqual(st.pos, length)
            events.append(("prefix", length))

        node = next(n for n in ast.parse(paths["decode"].read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_prefill_chunks")
        import time
        ns = {"time": time,
              "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
                  synchronize=lambda: events.append(("sync", st.pos)))),
              "stage": lambda w, buf, chunks: events.append(("stage", list(chunks[0][1]))),
              "compute": lambda *args, **kwargs: Tensor("logits"), "commit": commit,
              "mtp_forward": lambda *args: events.append(("mtp", None))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(paths["decode"]), "exec"), ns)
        self.assertEqual(ns["_prefill_chunks"](e, prompt, None, begin, None, use_mtp,
                                               None, None, None, on_chunk if callback else None,
                                               on_prefix if checkpoint else None), 77)
        self.assertEqual(st.pos, len(prompt))
        self.assertEqual(e.last_streams.value, "prefill")
        return events

    def test_yield_only_after_commit_and_not_after_final_chunk(self):
        events = self.run_loop([0, 1, 2, 3, 4])
        yields = [(i, v) for i, (kind, v) in enumerate(events) if kind == "decode"]
        self.assertEqual([v for i, v in yields], [2, 4])
        for i, value in yields:
            self.assertEqual(events[i - 1], ("sync", value))
            self.assertEqual(events[i - 2], ("commit", value))

    def test_cached_resume_and_no_mtp_keep_same_yield_boundaries(self):
        events = self.run_loop([0, 1, 2, 3, 4, 5, 6], begin=3, use_mtp=False)
        self.assertEqual([v for kind, v in events if kind == "decode"], [5])
        self.assertFalse(any(kind == "mtp" for kind, value in events))

    def test_single_chunk_and_synchronous_warm_need_no_callback(self):
        self.assertFalse(any(kind == "decode" for kind, _ in self.run_loop([0, 1])))
        self.assertFalse(any(kind == "decode" for kind, _ in self.run_loop([0, 1, 2], callback=False)))

    def test_exactly_one_checkpoint_before_final_chunk(self):
        events = self.run_loop(list(range(7)), checkpoint=True)
        self.assertEqual([v for kind, v in events if kind == "prefix"], [6])
        self.assertFalse(any(kind == "prefix" for kind, _ in self.run_loop([0, 1], checkpoint=True)))

    def test_checkpoint_mtp_excludes_the_unmatched_successor(self):
        cls = next(n for n in ast.parse(paths["multi"].read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
        admit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "admit")
        keep = next(n for n in admit.body if isinstance(n, ast.FunctionDef) and n.name == "keep_prefix")
        keep.body = [n for n in keep.body if not isinstance(n, ast.Nonlocal)]
        outer = ast.parse("def run(st, mtp, length, tail):\n    prefix = None\n").body[0]
        outer.body.extend(keep.body)
        outer.body.append(ast.Return(ast.Name(id="prefix", ctx=ast.Load())))
        module = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
        ns = {}
        exec(compile(module, str(paths["multi"]), "exec"), ns)
        st = types.SimpleNamespace(snapshot=lambda: {"pos": 4096, "mtp_len": 4096})
        length, snap, tail = ns["run"](st, True, 4096, Tensor("tail"))
        self.assertEqual(snap, {"pos": 4096, "mtp_len": 4095})
        self.assertEqual(tail.value, "tail")
        self.assertEqual(ns["run"](st, False, 4096, None)[1]["mtp_len"], 4096)

    def test_multi_decoder_forwards_hook_without_changing_admit_defaults(self):
        cls = next(n for n in ast.parse(paths["multi"].read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "admit_cooperatively")
        ns = {"Stream": Stream}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(paths["multi"]), "exec"), ns)
        calls = []
        target = types.SimpleNamespace(admit=lambda s, **kwargs: calls.append((s, kwargs)))
        stream, callback = object(), object()
        ns["admit_cooperatively"](target, stream, callback)
        self.assertEqual(calls, [(stream, {"on_chunk": callback})])


class PrefixRetentionTests(unittest.TestCase):
    """Exercise the real admission and slot-selection methods, not a cache helper."""

    def decoder(self, cached=8, *, checkpoint=None, mtp=True):
        cls = next(n for n in ast.parse(paths["multi"].read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
        methods = {"_busy", "_drop_kept", "_slot_for", "_remember", "admit"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        for method in cls.body:
            method.decorator_list = []
        import time
        snapshots = []
        st = types.SimpleNamespace(pos=cached, mtp_len=cached)

        def snapshot():
            snap = {"pos": st.pos, "mtp_len": st.mtp_len}
            snapshots.append(snap)
            return snap

        st.snapshot = snapshot
        tail = Tensor("original prefix tail") if mtp else None
        original = {"pos": cached, "mtp_len": cached - 1 if mtp else cached}

        def prefill(e, prompt, sampling, **kwargs):
            resume = kwargs["resume"]
            if resume is not None:
                self.assertEqual(resume["state"]["pos"], cached)
            if checkpoint is not None and kwargs["on_prefix"] is not None:
                st.pos = st.mtp_len = checkpoint
                kwargs["on_prefix"](checkpoint, Tensor("new prefix tail"))
            st.pos = st.mtp_len = len(prompt)
            e.last_streams = Tensor("prompt end")
            return 42

        ns = {"time": time, "Stream": Stream, "prefill": prefill,
              "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
              "_slot": lambda w, state, *args: types.SimpleNamespace(st=state)}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), cls], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(paths["multi"]), "exec"), ns)
        d = object.__new__(ns["MultiDecoder"])
        d.capacity, d.depth, d.keep = 100, 6 if mtp else 0, 8
        d.w = d.buf = d.pbuf = object()
        d.mbuf = object() if mtp else None
        d.vision = types.SimpleNamespace(encode=lambda *args: object())
        d.copy, d.eos, d.next_id = False, (), 0
        d.streams = {}
        d.free = [] if cached else [st]
        d.kept = [(list(range(cached)), st, original, tail)] if cached else []
        return d, original, tail, snapshots

    def test_repeated_tail_only_resumes_retain_original_checkpoint(self):
        d, original, tail, snapshots = self.decoder()
        for _ in range(4):
            stream = Stream(list(range(10)), 1)
            d.admit(stream)
            self.assertEqual(stream.cached, 8)
            self.assertEqual(d.kept[0][0], list(range(8)))
            self.assertIs(d.kept[0][2], original)
            self.assertIs(d.kept[0][3], tail)
            d.streams.pop(stream.sid)
        self.assertEqual(original, {"pos": 8, "mtp_len": 7})
        self.assertEqual(snapshots, [], "tail-only reuse must not allocate a new snapshot")

    def test_extended_resume_advances_checkpoint_and_adjusts_mtp_once(self):
        d, original, tail, snapshots = self.decoder(checkpoint=9)
        d.admit(Stream(list(range(10)), 1))
        self.assertEqual(d.kept[0][0], list(range(9)))
        self.assertEqual(d.kept[0][2], {"pos": 9, "mtp_len": 8})
        self.assertEqual(d.kept[0][3].value, "new prefix tail")
        self.assertEqual(original, {"pos": 8, "mtp_len": 7})
        self.assertEqual(len(snapshots), 1)

    def test_no_mtp_retains_checkpoint_without_successor_adjustment(self):
        d, original, tail, snapshots = self.decoder(mtp=False)
        d.admit(Stream(list(range(10)), 1))
        self.assertIs(d.kept[0][2], original)
        self.assertEqual(d.kept[0][2]["mtp_len"], 8)
        self.assertIsNone(d.kept[0][3])

    def test_cold_long_prompt_keeps_new_checkpoint(self):
        d, _, _, snapshots = self.decoder(cached=0, checkpoint=8)
        d.admit(Stream(list(range(10)), 1))
        self.assertEqual(d.kept[0][0], list(range(8)))
        self.assertEqual(d.kept[0][2], {"pos": 8, "mtp_len": 7})
        self.assertEqual(len(snapshots), 1)

    def test_cold_short_prompt_keeps_existing_prompt_end_behavior(self):
        d, _, _, snapshots = self.decoder(cached=0)
        d.admit(Stream([0, 1], 1))
        self.assertEqual(d.kept[0][0], [0, 1])
        self.assertEqual(d.kept[0][2], {"pos": 2, "mtp_len": 2})
        self.assertEqual(d.kept[0][3].value, "prompt end")

    def test_image_and_no_draft_requests_never_keep_a_prefix(self):
        for kwargs in ({"draft": False}, {"vision": object()}):
            d, _, _, snapshots = self.decoder(cached=0, checkpoint=8)
            d.admit(Stream(list(range(10)), 1, **kwargs))
            self.assertEqual(d.kept, [])
            self.assertEqual(snapshots, [])


if __name__ == "__main__":
    unittest.main()
