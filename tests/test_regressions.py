import os
import unittest
from unittest.mock import patch

import torch

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import SFTDataset


class ModelRegressionTests(unittest.TestCase):
    def test_checkpointed_and_regular_training_have_matching_gradients(self):
        torch.manual_seed(7)
        regular_cfg = ModelConfig.create_micro()
        regular_cfg.gradient_checkpointing = False
        checkpointed_cfg = ModelConfig.create_micro()
        checkpointed_cfg.gradient_checkpointing = True

        regular = Axelion(regular_cfg)
        checkpointed = Axelion(checkpointed_cfg)
        checkpointed.load_state_dict(regular.state_dict())
        regular.train()
        checkpointed.train()
        inputs = torch.randint(0, regular_cfg.vocab_size, (2, 32))
        targets = torch.roll(inputs, shifts=-1, dims=1)

        _, regular_loss, _ = regular(inputs, targets=targets)
        _, checkpointed_loss, _ = checkpointed(inputs, targets=targets)
        regular_loss.backward()
        checkpointed_loss.backward()

        self.assertTrue(torch.allclose(regular_loss, checkpointed_loss, atol=1e-6))
        for first, second in zip(regular.parameters(), checkpointed.parameters()):
            self.assertIsNotNone(first.grad)
            self.assertTrue(torch.allclose(first.grad, second.grad, atol=1e-5, rtol=1e-4))

    def test_all_ignored_targets_fail_loudly(self):
        model = Axelion(ModelConfig.create_micro())
        inputs = torch.zeros((1, 8), dtype=torch.long)
        targets = torch.full_like(inputs, -100)
        with self.assertRaises(ValueError):
            model(inputs, targets=targets)


class DatasetRegressionTests(unittest.TestCase):
    def test_sft_dataset_masks_prompt_and_keeps_assistant(self):
        tokenizer = AxelionTokenizer()
        path = os.path.join(os.path.dirname(__file__), "tmp_sft.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"messages":[{"role":"user","content":"question"},'
                    '{"role":"assistant","content":"answer"}]}\n'
                )
            dataset = SFTDataset(path, tokenizer, max_seq_len=128)
            _, labels = dataset[0]
            self.assertTrue(torch.any(labels == -100))
            self.assertTrue(torch.any(labels != -100))
        finally:
            if os.path.exists(path):
                os.remove(path)


class AuthenticationRegressionTests(unittest.TestCase):
    def test_client_uses_bearer_scheme(self):
        from axelion.client import AxelionClient

        client = AxelionClient(api_key="secret")
        with patch("axelion.client.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b"{}"
            client.health()
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")


if __name__ == "__main__":
    unittest.main()
