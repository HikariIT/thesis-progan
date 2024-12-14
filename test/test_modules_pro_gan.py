import torch
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from net.modules.pro_gan.weighted_modules import WeightedConv2d, WeightedLinear
from net.modules.pro_gan.pixelwise_norm import PixelwiseNormalization
from net.modules.pro_gan.minibatch_stddev import MiniBatchStdDev
from net.modules.pro_gan.upsample import Upsample


class TestModulesProGAN:

    def test_MiniBatchStdDev(self):
        minibatch_stddev = MiniBatchStdDev().cuda()

        random_tensor = torch.zeros(32, 512, 4, 4).cuda()
        output = minibatch_stddev(random_tensor)

        assert output.size() == (32, 513, 4, 4)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 512 * 1e-4, rel_tol=1e-4)

    def test_MiniBatchStdDev_custom_group_size(self):
        minibatch_stddev = MiniBatchStdDev(8).cuda()

        random_tensor = torch.zeros(32, 512, 4, 4).cuda()
        output = minibatch_stddev(random_tensor)

        assert output.size() == (32, 513, 4, 4)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 512 * 1e-4, rel_tol=1e-4)

    def test_MiniBatchStdDev_single_group(self):
        minibatch_stddev = MiniBatchStdDev(32).cuda()

        random_tensor = torch.zeros(32, 512, 4, 4).cuda()
        output = minibatch_stddev(random_tensor)

        assert output.size() == (32, 513, 4, 4)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 512 * 1e-4, rel_tol=1e-4)

    def test_PixelwiseNormalization(self):
        pixelwise_norm = PixelwiseNormalization().cuda()

        random_tensor = (torch.ones(32, 64, 4, 4) * 2.0).cuda()
        output = pixelwise_norm(random_tensor)

        # Tensor should be normalized by its mean, so here by a factor of 2
        expected_sum = torch.sum(random_tensor).item() / 2.0

        assert output.size() == (32, 64, 4, 4)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), expected_sum, rel_tol=1e-4)

    def test_Upsample(self):
        upsample = Upsample().cuda()

        random_tensor = torch.zeros(32, 64, 4, 4).cuda()
        output = upsample(random_tensor)

        assert output.size() == (32, 64, 8, 8)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 0.0, rel_tol=1e-4)

    def test_WeightedConv2d(self):
        weighted_conv = WeightedConv2d(3, 64, 3, 1, 1).cuda()

        random_tensor = torch.zeros(32, 3, 4, 4).cuda()
        output = weighted_conv(random_tensor)

        assert output.size() == (32, 64, 4, 4)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 0.0, rel_tol=1e-4)

    def test_WeightedLinear(self):
        weighted_linear = WeightedLinear(512, 256).cuda()

        random_tensor = torch.zeros(32, 512).cuda()
        output = weighted_linear(random_tensor)

        assert output.size() == (32, 256)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 0.0, rel_tol=1e-4)

    def test_WeightedLinear_no_bias(self):
        weighted_linear = WeightedLinear(512, 256).cuda()
        weighted_linear.linear.bias = None # type: ignore

        random_tensor = torch.zeros(32, 512).cuda()
        output = weighted_linear(random_tensor)

        assert output.size() == (32, 256)
        assert output.device == torch.device('cuda:0')
        assert math.isclose(torch.sum(output).item(), 0.0, rel_tol=1e-4)