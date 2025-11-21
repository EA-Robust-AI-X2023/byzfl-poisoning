from loguru import logger
import numpy as np
import torch as torch


class StaticLabelFlipping(object): 
    """
    Description
    -----------

    Flip target labels with a fixed permutation.

    Initialization parameters
    --------------------------

    permutation: dict
        A dictionary that maps a label to its flipped version.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    model : torch.nn.Module
        The local model.
    inputs : numpy.ndarray or torch.Tensor
        A collection of input features.
    targets : numpy.ndarray or torch.Tensor
        A collection of targets.

    Returns
    -------
    tuple
        The inputs and targets with flipped labels.
    """
    def __init__(self, permutation = {0:9, 1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1, 9:0}):
        if not isinstance(permutation, dict):
            raise TypeError("`permutation` must be a dict.")
        self.permutation = {int(i): int(j) for i, j in permutation.items()}
        #convert to list for simplicity of use
        self.permutation = [self.permutation[i] for i in range(len(self.permutation))]

    
    def __call__(self, model, inputs, targets):
        if torch.is_tensor(targets):
            
            perm = torch.tensor(self.permutation, device=targets.device)
            poisoned_targets = perm[targets]
            
            return inputs, poisoned_targets
        else:
            poisoned_targets = np.array([self.permutation[int(t)] for t in targets], dtype=np.int64)
            return inputs, poisoned_targets
    
    
class DynamicLabelFlipping(object):
    """
    Description
    -----------

    Flip target labels towards the least likely label (according to the model and current parameters)


    Calling the instance
    --------------------

    Input parameters
    ----------------

    model : torch.nn.Module
        The local model.
    inputs : numpy.ndarray or torch.Tensor
        A collection of input features.
    targets : numpy.ndarray or torch.Tensor
        A collection of targets.

    Returns
    -------
    tuple
        The inputs and targets with flipped labels.
    """
    
    def __init__(self, print_flips=False):
        self.print_flips = print_flips
    
    def __call__(self, model, inputs, targets):
        flipped_targets = model(inputs).argmin(dim=1)

        if self.print_flips:
            logger.debug("Flipping targets:")
            for i, (original, flipped) in enumerate(zip(targets, flipped_targets)):
                if original != flipped:
                    logger.debug(f"{original} -> {flipped} /")
            logger.debug("\n")

        return inputs, flipped_targets
    
    
class NoAttack(object):
    """
    Description
    -----------

    No attack is applied, and the original inputs and targets are returned.
    """
    def __init(self):
        pass
    
    def __call__(self, model, inputs, targets):
        return inputs, targets
    
