# import sys

# print(sys.path)
import numpy as np

from experts.cartoon_expert import CartoonWB
from experts.depth_expert import DepthModelXTC
from experts.edges_expert import EdgesModel
from experts.grayscale_expert import Grayscale
from experts.halftone_expert import HalftoneModel
from experts.hsv_expert import HSVExpert
from experts.normals_expert import SurfaceNormalsXTC
from experts.rgb_expert import RGBModel
from experts.semantic_segmentation_expert import SSegHRNet, SSegHRNet_v2
from experts.sobel_expert import (SobelEdgesExpertSigmaLarge,
                                  SobelEdgesExpertSigmaMedium,
                                  SobelEdgesExpertSigmaSmall)
from experts.superpixel_expert import SuperPixel


class Experts:
    def __init__(self, dataset_name, full_experts=True, selector_map=None):
        self.methods = [
            RGBModel(full_experts),
            HalftoneModel(full_experts, 0),
            Grayscale(full_experts),
            HSVExpert(full_experts),
            DepthModelXTC(full_experts),
            SurfaceNormalsXTC(dataset_name=dataset_name,
                              full_expert=full_experts),
            SobelEdgesExpertSigmaSmall(full_experts),
            SobelEdgesExpertSigmaMedium(full_experts),
            SobelEdgesExpertSigmaLarge(full_experts),
            EdgesModel(full_experts),
            SuperPixel(full_experts),
            CartoonWB(full_experts),
            SSegHRNet(dataset_name=dataset_name, full_expert=full_experts)
        ]

        if selector_map is None:
            selector_map = np.arange(len(self.methods))

        self.methods = np.array(self.methods)[selector_map].tolist()

        print("==================")
        print("USED", len(self.methods), "EXPERTS:",
              [method.__class__.__name__ for method in self.methods])
        print("==================")
