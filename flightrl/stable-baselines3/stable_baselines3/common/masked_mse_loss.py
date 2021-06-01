import torch


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        # mask_bool = torch.logical_and(torch.flatten(mask) == 1, torch.flatten(target) != 0)
        mask_bool = torch.flatten(mask) == 1
        
        diff2 = torch.masked_select((torch.flatten(input) - torch.flatten(target)) ** 2.0, mask_bool)
        if torch.sum(mask_bool) != 0:
            result = torch.sum(diff2) / torch.sum(mask_bool)
        else:
            result = torch.sum(diff2)*0

        return result

class MaskedABSLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedABSLoss, self).__init__()

    def forward(self, input, target, mask):
        # mask_bool = torch.logical_and(torch.flatten(mask) == 1, torch.flatten(target) != 0)
        mask_bool = torch.flatten(mask) == 1
        
        diff2 = torch.masked_select(torch.abs(torch.flatten(input) - torch.flatten(target)), mask_bool)
        if torch.sum(mask_bool) != 0:
            result = torch.sum(diff2) / torch.sum(mask_bool)
        else:
            result = torch.sum(diff2)*0

        return result
