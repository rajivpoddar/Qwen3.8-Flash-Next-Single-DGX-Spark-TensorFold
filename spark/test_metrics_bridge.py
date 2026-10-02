import unittest

from metrics_bridge import LiveCounters, translate, translate_native


def health(prompt=65, completion=10, running=1):
    return dict(prompt_tokens_total=prompt, completion_tokens_total=completion,
                requests_running=running)


class TranslateTests(unittest.TestCase):
    def test_live_tokens_advance_before_gateway_request_finishes(self):
        raw = 'litellm_output_tokens_metric_total{model="qwen"} 999999\n'
        before = translate(raw, health(completion=10))
        during = translate(raw, health(completion=16))
        self.assertIn("vllm_generation_tokens_total 10\n", before)
        self.assertIn("vllm_generation_tokens_total 16\n", during)
        self.assertNotIn("999999", during)

    def test_gateway_completion_does_not_create_throughput_burst(self):
        before = translate('litellm_output_tokens_metric_total 0\n', health(completion=1024))
        after = translate('litellm_output_tokens_metric_total 1024\n', health(completion=1024))
        self.assertEqual(before, after)

    def test_only_measured_e2e_latency_and_histogram_are_mapped(self):
        raw = '''litellm_request_total_latency_metric_sum{model="qwen3.8-flash-next",user="a"} 10
litellm_request_total_latency_metric_sum{model="qwen3.8-flash-next",user="b"} 30
litellm_request_total_latency_metric_count{model="qwen3.8-flash-next",user="a"} 1
litellm_request_total_latency_metric_count{model="qwen3.8-flash-next",user="b"} 2
litellm_request_total_latency_metric_bucket{model="qwen3.8-flash-next",le="1.0",user="a"} 0
litellm_request_total_latency_metric_bucket{model="qwen3.8-flash-next",le="+Inf",user="a"} 1
litellm_request_total_latency_metric_bucket{model="qwen3.8-flash-next",le="+Inf",user="b"} 2
litellm_request_total_latency_metric_sum{model="another-model"} 5000
litellm_llm_api_time_to_first_token_metric_sum 0.13
litellm_llm_api_time_to_first_token_metric_count 1
litellm_deployment_latency_per_output_token_sum 0.01
litellm_in_flight_requests 4
'''
        result = translate(raw, health(running=0))
        self.assertIn("# TYPE vllm_e2e_request_latency_seconds histogram\n", result)
        self.assertIn("vllm_e2e_request_latency_seconds_sum 40\n", result)
        self.assertIn("vllm_e2e_request_latency_seconds_count 3\n", result)
        self.assertIn('vllm_e2e_request_latency_seconds_bucket{le="+Inf"} 3\n', result)
        self.assertIn("vllm_num_requests_running 0\n", result)
        for unsupported in ("time_to_first_token", "queue_time", "inter_token_latency",
                            "time_per_output_token", "kv_cache", "batch"):
            self.assertNotIn(unsupported, result)
        self.assertNotIn("user=", result)

    def test_nonfinite_and_unrelated_gateway_samples_are_dropped(self):
        result = translate('litellm_request_total_latency_metric_sum NaN\nother_total 5\n', health())
        self.assertNotIn("latency_seconds", result)
        self.assertNotIn("other_total", result)

    def test_does_not_round_large_live_token_counts(self):
        self.assertIn("vllm_generation_tokens_total 123456789\n",
                      translate('', health(completion=123456789)))

    def test_rejects_missing_and_invalid_engine_metrics(self):
        for key in health():
            for value in (-1, float('nan'), float('inf'), True, 1.5, "1", None):
                invalid = health()
                invalid[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    translate('', invalid)
            invalid = health()
            del invalid[key]
            with self.assertRaises(KeyError):
                translate('', invalid)


class LiveCounterTests(unittest.TestCase):
    def test_seed_preserves_existing_totals_over_gateway_replacement(self):
        counters = LiveCounters()
        counters.seed({'previous': {'prompt_tokens_total': 100, 'completion_tokens_total': 10},
                       'totals': {'prompt_tokens_total': 1000, 'completion_tokens_total': 200}})
        observed = counters.observe(health(110, 12))
        self.assertEqual(observed['prompt_tokens_total'], 1010)
        self.assertEqual(observed['completion_tokens_total'], 202)

    def test_native_metrics_preserve_real_ttft_and_live_tokens(self):
        raw = '''# TYPE tensorfold:time_to_first_token_seconds histogram
tensorfold:time_to_first_token_seconds_sum 20
tensorfold:time_to_first_token_seconds_count 2
tensorfold:time_to_first_token_seconds_bucket{le="+Inf"} 2
tensorfold:e2e_request_latency_seconds_sum 30
tensorfold:e2e_request_latency_seconds_count 2
tensorfold:generation_tokens_total 999999
tensorfold:request_latency_seconds_sum 30
'''
        result = translate_native(raw, health(completion=12))
        self.assertIn('vllm_time_to_first_token_seconds_sum 20', result)
        self.assertIn('vllm_e2e_request_latency_seconds_sum 30', result)
        self.assertIn('vllm_generation_tokens_total 12', result)
        self.assertNotIn('999999', result)
        self.assertNotIn('vllm_request_latency_seconds_sum', result)

    def test_model_reset_does_not_make_dashboard_rates_negative(self):
        counters = LiveCounters()
        self.assertEqual(counters.observe(health(100, 10))["completion_tokens_total"], 10)
        self.assertEqual(counters.observe(health(100, 15))["completion_tokens_total"], 15)
        reset = counters.observe(health(5, 2))
        self.assertEqual(reset["prompt_tokens_total"], 105)
        self.assertEqual(reset["completion_tokens_total"], 17)
        self.assertEqual(counters.observe(health(5, 6))["completion_tokens_total"], 21)

    def test_invalid_snapshot_cannot_partially_advance_counters(self):
        counters = LiveCounters()
        counters.observe(health(100, 10))
        with self.assertRaises(ValueError):
            counters.observe(health(200, -1))
        self.assertEqual(counters.observe(health(210, 12))["prompt_tokens_total"], 210)

    def test_zero_during_idle_is_not_a_reset(self):
        counters = LiveCounters()
        counters.observe(health(0, 0, 0))
        counters.observe(health(0, 0, 0))
        self.assertEqual(counters.observe(health(100, 25))["completion_tokens_total"], 25)


if __name__ == "__main__":
    unittest.main()
