import torch

def init_ema_model(model, model_ema, device=None):
    with torch.no_grad():
        for (mp, ep) in zip(model.parameters(), model_ema.parameters()):
            data = mp.data
            if device is not None:
                data = data.to(device)
            ep.data.copy_(data)

def moving_average(model, model_ema, beta=0.999, device=None):
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            data = param.data
            if device is not None:
                data = data.to(device)
            param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
