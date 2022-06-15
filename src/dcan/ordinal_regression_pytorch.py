from torch import nn


# https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99
def ordinal_regression(predictions: List[List[float]], targets: List[float]):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    return nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)
