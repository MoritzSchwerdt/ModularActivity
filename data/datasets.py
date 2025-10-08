import numpy as np
import torch
from torchvision.datasets import STL10
from .augmentations import contrast_transforms

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

def get_dataloaders(dataset_path, batch_size, contrast_transforms, num_workers=0):
    unlabeled_data = STL10(
        root=dataset_path, split='unlabeled', download=True,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2)
    )
    train_data_contrast = STL10(
        root=dataset_path, split='train', download=True,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2)
    )

    def numpy_collate_contrastive(batch):
        imgs1, imgs2 = [[b[0][i] for b in batch] for i in range(2)]
        return np.stack(imgs1 + imgs2, axis=0)

    train_loader = torch.utils.data.DataLoader(
        unlabeled_data, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=numpy_collate_contrastive,
        num_workers=num_workers, persistent_workers=False,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = torch.utils.data.DataLoader(
        train_data_contrast, batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=numpy_collate_contrastive,
        num_workers=num_workers, persistent_workers=False
    )
    return train_loader, val_loader
