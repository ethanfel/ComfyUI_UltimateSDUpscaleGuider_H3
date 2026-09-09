"""CPU precision integration: COMFYUI_ROOT=/path/to/ComfyUI python -B test/precision_unit_test.py."""

import contextlib
import io
import unittest
from unittest.mock import patch

import video_unit_test as env
import torch
import numpy as np

usdu = env.usdu
utils = usdu.usdu.Script.run.__globals__['usdu_utils']


class PrecisionTests(unittest.TestCase):
    def test_rounding_clamping_and_tensor_contract(self):
        source = torch.linspace(0, 1, 4096).reshape(1, 32, 128, 1).expand(-1, -1, -1, 3).clone()
        for precision, dtype in (('8-bit', np.uint8), ('16-bit', np.uint16)):
            frame = utils.tensor_to_frame(source, precision=precision)
            pixels = utils.frame_pixels(frame)
            self.assertEqual(pixels.dtype, dtype)
            restored = utils.pil_batch_to_tensor([frame, frame])
            self.assertEqual(restored.dtype, torch.float32)
            self.assertEqual(restored.device.type, 'cpu')
            self.assertEqual(tuple(restored.shape), (2, 32, 128, 3))
            self.assertLessEqual((restored - source).abs().max(), 1 / 255 if precision == '8-bit' else 0.51 / 65535)
            self.assertEqual(len(torch.unique(restored)), 256 if precision == '8-bit' else 4096)
        invalid = torch.tensor([[[[-1., 2., float('nan')], [float('inf'), -float('inf'), 0.5]]]])
        pixels = utils.tensor_to_frame(invalid, precision='16-bit').pixels
        np.testing.assert_array_equal(pixels, [[[0, 65535, 0], [65535, 0, 32768]]])

    def test_decode_retains_single_uint16_level_changes(self):
        source = torch.full((5, 64, 128, 3), 30000 / 65535)
        seen = []

        def encode(vae, images):
            seen.append((images.dtype, tuple(images.shape)))
            return ({'samples': images.clone()},)

        def sample(guider, seed, sampler, sigmas, latent):
            return {'samples': latent['samples'] + 1 / 65535}

        def decode(vae, latent):
            return (latent['samples'],)

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(env.processing.VAEEncode, 'encode', side_effect=encode), \
                patch.object(env.processing, 'sample_with_guider', new=sample), \
                patch.object(env.processing.VAEDecode, 'decode', side_effect=decode):
            for precision in ('8-bit', '16-bit'):
                result = usdu.UltimateSDUpscaleNoUpscaleGuider().upscale(source, **env.settings(
                    canvas_precision=precision, mode_type='Linear'))[0]
                expected = 30001 / 65535 if precision == '16-bit' else 116 / 255
                self.assertTrue(torch.equal(result, torch.full_like(result, expected)))
        self.assertEqual(seen, [(torch.float32, (5, 64, 64, 3))] * 4)

    def test_initial_upscale_keeps_uint16_levels(self):
        source = torch.linspace(0, 1, 64 * 128).reshape(1, 64, 128, 1).expand(-1, -1, -1, 3)
        model_node = usdu.UpscalerData().scaler.upscale.__globals__['ImageUpscaleWithModel']

        def execute(model, images):
            self.assertIs(images, source)
            return (images.repeat_interleave(2, 1).repeat_interleave(2, 2),)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(model_node, 'execute', side_effect=execute):
            for model in (None, object()):
                result = usdu.UltimateSDUpscaleGuider().upscale(source, upscale_by=2, upscale_model=model,
                    **env.settings(canvas_precision='16-bit'))[0]
                self.assertEqual(tuple(result.shape), (1, 128, 256, 3))
                self.assertGreater(len(torch.unique(result)), 4096)
                self.assertEqual(usdu.shared.canvas_precision, '8-bit')


if __name__ == '__main__':
    unittest.main(argv=[__file__])
