import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(
    est_time=200,
    suite="stage-b-test-2-gpu-large",
)

MODEL = "mistralai/Mistral-Small-4-119B-2603"


class TestMistralSmall4TextOnly(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.6
    model = MODEL
    other_args = ["--tp-size", "2", "--trust-remote-code"]


if __name__ == "__main__":
    unittest.main()
