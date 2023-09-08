import torch

class RandomSampleFeaturesTransform:
    """
    When the features tensor has > 1 sample, randomly get the features of one sample
    """
    def __call__(self, features):
        if features.size(0) > 1:
            features = features[torch.randint(features.size(0), (1,))]
        return features

class AvgFeaturesTransform:
    """
    When the features tensor has > 1 sample, average the features of all samples
    """
    def __call__(self, features):
        if features.size(0) > 1:
            features = features.mean(dim=0, keepdim=True)
        return features
    

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, features):
        if torch.rand(1).item() < self.p:
            return self.transforms1(features)
        return self.transforms2(features)
    
class UnNormalize(object):
    """
    Unnormalize a tensor image with mean and standard deviation
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor