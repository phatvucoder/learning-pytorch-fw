import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # cd to the directory of this file

import torch
from torchvision import models

from utils.image_transform import ImageTransform
from utils.config import *
from utils.utils import *
from utils.mydataset import MyDataset
from torch.utils import data

def main():
    # Create a directory to save weights
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    # Delete the last and best weights if they exist
    for file in os.listdir(weight_dir):
        if file == last_weights or file == best_weights:
            os.remove(os.path.join(weight_dir, file))


    # Load and preprocess data
    train_list = make_data_list("train")
    val_list = make_data_list("val")

    train_dataset = MyDataset(file_list=train_list, transform=ImageTransform(resize=resize, mean=mean, std=std), phase='train')
    val_dataset = MyDataset(file_list=val_list, transform=ImageTransform(resize=resize, mean=mean, std=std), phase='val')

    # Create dataloader
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # Create model
    net = models.vgg16(pretrained=use_pretrained)

    # Update the final layer
    net = update_out_features(net, num_classes)

    # Set the model to training mode
    net = net.train()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer setting
    params_to_update = get_transfer_learning_vgg_params(net)
    mode = "fine_tuning" # "fine_tuning"  or "transfer_learning"
    if mode == "transfer_learning":
        print("Starting transfer learning mode for the final layer...")
        params_to_update = get_transfer_learning_vgg_params(net)
        # params_to_update2 = get_transfer_learning_params(net)
        # assert params_to_update == params_to_update2
        optimizer = torch.optim.SGD(params=params_to_update, lr=tl_lr, momentum=momentum)
    elif mode == "fine_tuning":
        print("Starting fine-tuning mode for all network parameters...")
        params_to_update1, params_to_update2, params_to_update3 = get_fine_tuning_vgg_params(net)
        # params_to_update1a, params_to_update2a, params_to_update3a = get_fine_tuning_params(net)
        # assert params_to_update1 == params_to_update1a
        # assert params_to_update2 == params_to_update2a
        # assert params_to_update3 == params_to_update3a
        optimizer = torch.optim.SGD([
            {"params": params_to_update1, "lr": ft_lr[0]},
            {"params": params_to_update2, "lr": ft_lr[1]},
            {"params": params_to_update3, "lr": ft_lr[2]}
        ], momentum=momentum)
    else:
        print("Invalid mode")
        exit()

    # get device
    device = get_device()
    print(f"Device: {device}")

    # Train the model
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, device=device)
    print("Finished training")

    # Load the best model
    best_model_path = os.path.join(weight_dir, best_weights)
    net = load_model(net, best_model_path)
    print(net)

if __name__ == "__main__":
    main()