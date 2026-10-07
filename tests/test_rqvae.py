from __future__ import annotations

import unittest

import torch

from generative.rqvae import RQVAE


class RQVAETests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = RQVAE(input_dim=8, hidden_dim=16, latent_dim=8, num_codebooks=3, codebook_size=4)
        self.x = torch.randn(7, 8)

    def test_forward_and_quantizer_shapes(self) -> None:
        out = self.model(self.x)
        self.assertEqual(tuple(self.model.encoder(self.x).shape), (7, 8))
        self.assertEqual(tuple(out.quantized.shape), (7, 8))
        self.assertEqual(tuple(out.reconstruction.shape), (7, 8))
        self.assertEqual(tuple(out.code_ids.shape), (7, 3))
        self.assertEqual(len(out.residuals), 3)
        self.assertTrue(all(tuple(value.shape) == (7, 8) for value in out.residuals))

    def test_ste_and_codebook_gradient_paths(self) -> None:
        out = self.model(self.x)
        out.total_loss.backward()
        self.assertIsNotNone(self.model.encoder[0].weight.grad)
        self.assertGreater(float(self.model.encoder[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.model.quantizer.codebooks.grad)
        self.assertGreater(float(self.model.quantizer.codebooks.grad.abs().sum()), 0.0)

    def test_loss_is_finite_and_no_nan(self) -> None:
        out = self.model(self.x)
        self.assertTrue(torch.isfinite(out.total_loss))
        self.assertTrue(torch.isfinite(out.reconstruction).all())
        self.assertTrue(torch.isfinite(out.quantized).all())


if __name__ == "__main__":
    unittest.main()

