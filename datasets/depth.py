import h5py
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import utils


class DepthDataset(Dataset):
    def __init__(self, file_paths, num_views=30, train=False):
        super().__init__()

        self.file_paths = file_paths
        self.num_views = num_views
        self.length = len(file_paths)
        self.is_train = train

        self.transform_list = [
            "identity",
            "jitter",
            "gray",
            "equalize",
            "posterize",
            "solarize",
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx // self.num_views

        view_idx = np.random.randint(0, self.num_views)
        with h5py.File(self.file_paths[idx], "r") as hf:
            image = hf["rgb"][view_idx] / 255.0
            depth = hf["depth"][view_idx]
            mask = hf["mask"][view_idx]
            K = hf["K"][:]
            cam2world = hf["cam2world"][view_idx]

        depth[depth == 10.0] = 0.0

        # if self.is_train:
        # # Bounding box computation and move bbox
        # x, y = np.where(mask)
        # bbox = min(x), max(x), min(y), max(y)
        # W, H = 256, 256
        # w, h = bbox[1] - bbox[0], bbox[3] - bbox[2]

        # prob = np.random.rand()
        # if prob > 0.5:
        # min_range = min(1.0, 40 / (min(h, w)))
        # max_range = max(min_range, (256 * 0.9) / (max(h, w)))

        # scale_factor = np.random.uniform(min_range, max_range)
        # scale_factor = int(scale_factor * W) / W
        # image = utils.resize_array(image, bbox, scale_factor, pad_value=1.0)
        # depth = utils.resize_array(
        # depth, bbox, scale_factor, pad_value=10.0, inter="nearest"
        # )
        # mask = utils.resize_array(
        # mask, bbox, scale_factor, pad_value=0.0, inter="nearest"
        # )
        # # depth = depth / scale_factor
        # depth[np.where(mask == 0)] = 10.0
        # image[np.where(mask == 0)] = 1.0

        image = T.ToTensor()(image)
        depth = torch.tensor(depth).float()
        mask = torch.tensor(mask).float()

        # Image augmentations
        if self.is_train:
            probs = torch.ones(len(self.transform_list))
            dist = torch.distributions.categorical.Categorical(probs=probs)
            aug = self.transform_list[dist.sample()]

            if aug == "jitter":
                image = T.ColorJitter(
                    brightness=(0.25, 0.75), hue=(-0.4, 0.4), saturation=(0.25, 0.75)
                )(image)
            if aug == "gray":
                image = T.Grayscale(3)(image)
            if aug == "equalize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomEqualize(1.0)(image)
                image = (image / 255.0).float()
            if aug == "posterize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomPosterize(bits=2, p=1.0)(image)
                image = (image / 255.0).float()
            if aug == "solarize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomSolarize(threshold=192, p=1.0)(image)
                image = (image / 255.0).float()

        image = image.permute(1, 2, 0).clone()
        image[torch.where(mask == 0)] = 1.0

        return {
            "images": image.numpy(),
            "depths": depth.numpy(),
            "masks": mask.numpy(),
            "K": torch.tensor(K).float(),
            "cam2world": torch.tensor(cam2world).float().numpy(),
        }
