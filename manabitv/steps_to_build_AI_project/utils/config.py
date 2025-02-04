import torch

torch.manual_seed(0)

use_pretrained = True
num_classes = 2

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

batch_size = 32
num_epochs = 10

tl_lr = 0.001
ft_lr = (1e-4, 5e-4, 1e-3)
momentum = 0.9

weight_dir = "./weights"
last_weights = "last_weights.pth"
best_weights = "best_weights.pth"