import torch

# model,x0,t,noise,beta
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, condition, condition_idx):
    # alpha cumprod
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # get xt
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # predict noise from xt and t
    output = model(x, condition, condition_idx, t.float())
    return (e - output).square().sum(dim=(1, 2, 3))



