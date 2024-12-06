"""
LSUV Weight Initialization
"""
# https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
# https://arxiv.org/abs/1511.06422
# https://github.com/fastai/course22p2/blob/master/nbs/11_initializing.ipynb

from torch.nn import init
from torch.utils.data import DataLoader

def lsuv_init(model, dataloader, tol=1e-5, max_iter=10):

    activations_stats = {}

    def set_weights(m):
        if hasattr(m, 'weight') and m.weight.ndimension() >= 2:
            init.orthogonal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


    def get_activations_hook(module, input, output):
        # if not hasattr(module, 'activations_stats'):
        #     module.activations_stats = {}
        
        mean = output.detach().data.mean()
        sd = output.detach().data.std()
        if not mean or not sd:
            print(f"Error in calculating mean or sd in: {module}")

        # print(f"{str(module)}: {mean}, {sd}")
        # module.activations_stats['mean'] = mean
        # module.activations_stats['std'] = sd
        activations_stats[module] = {}
        activations_stats[module]["mean"] = mean
        activations_stats[module]["sd"] = sd

        # activations_stats['mean'] = mean
        # activations_stats['std'] = sd

    model.apply(set_weights)

    single_batch = next(iter(dataloader))
    for layer in model.modules():
        if str(layer).startswith('NonDynamicallyQuantizableLinear'):
            continue
        if str(layer).startswith('Linear(in_features=1024, out_features=4096, bias=True)'):
            print("found")
        if hasattr(layer, 'weight'):
            hook = layer.register_forward_hook(get_activations_hook)
            try:
                for iteration in range(max_iter):
                    # print(f"Iteration {iteration}")
                    model(single_batch)  # Run model with a batch of data
                    
                    std = activations_stats[layer]["sd"]
                    mean = activations_stats[layer]["mean"] 
                    if abs(std - 1) < tol and abs(mean) < tol:
                        break
                    layer.weight.data /= std
                    layer.bias.data -= mean
            except KeyError:
                print("Error in layer: ", layer)
            hook.remove()

# Example usage:
# Assuming `model` is your PyTorch model and `train_loader` is your DataLoader
# lsuv_init(model, train_loader)