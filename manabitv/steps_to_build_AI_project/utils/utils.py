import glob
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.config import weight_dir, last_weights, best_weights

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

def make_data_list(phase="train"):
    root_path = "./data/hymenoptera_data"
    target_path = osp.join(root_path, phase)
    
    return glob.glob(osp.join(target_path, "*/*.jpg"))

def update_out_features(model, num_classes):
    if hasattr(model, "classifier"):  # VGG, DenseNet, EfficientNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):  # ViT
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)

    return model

def train_model(net, dataloader_dict, criterion, optimizer, num_epochs, weight_dir=weight_dir, last_weights=last_weights, best_weights=best_weights, device=None):
    if device is None:
        device = torch.device("cpu")

    best_acc = 0.0
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-------------")

        net.to(device)

        # Speeds up training if input image size is constant
        # Set to False if input size varies
        torch.backends.cudnn.benchmark = True
        
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"):
                continue
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.float() / len(dataloader_dict[phase].dataset) # mps does not support float64(double)
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_weights_path = osp.join(weight_dir, best_weights)
                torch.save(net.state_dict(), best_weights_path)

        last_weights_path = osp.join(weight_dir, last_weights)
        torch.save(net.state_dict(), last_weights_path)

def load_model(model, model_path):
    weights = torch.load(model_path)
    model.load_state_dict(weights)

    # # Load model trained on GPU to CPU
    # load_weights = torch.load(model_path, map_location=("gpu:0", "cpu"))
    # model.load_state_dict(load_weights)

    return model
    

def get_transfer_learning_vgg_params(model):
    params_to_update = []
    update_params_name = ["classifier.6.weight", "classifier.6.bias"]
    
    for name, param in model.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    return params_to_update

def get_fine_tuning_vgg_params(model):
    params_to_update1 = [] # features
    params_to_update2 = [] # hidden layers
    params_to_update3 = [] # output layer
    
    for name, param in model.named_parameters():
        if name.startswith("features"):
            param.requires_grad = True
            params_to_update1.append(param)
        elif name in ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]:
            param.requires_grad = True
            params_to_update2.append(param)
        elif name in ["classifier.6.weight", "classifier.6.bias"]:
            param.requires_grad = True
            params_to_update3.append(param)
        else:
            param.requires_grad = False

    return params_to_update1, params_to_update2, params_to_update3

def get_transfer_learning_params(model):
    params_to_update = []
    update_param_names = set()

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        classifier_modules = list(model.classifier.children())
        if len(classifier_modules) > 0:
            last_module = classifier_modules[-1]
            for name, param in model.classifier.named_parameters():
                if isinstance(last_module, nn.Linear):
                    if (param.ndim == 2 and param.shape == last_module.weight.shape) or \
                       (param.ndim == 1 and param.shape == last_module.bias.shape):
                        update_param_names.add("classifier." + name)
    elif hasattr(model, "fc"):
        update_param_names = {"fc.weight", "fc.bias"}
    elif hasattr(model, "head"):
        update_param_names = {"head.weight", "head.bias"}
    
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    return params_to_update


def get_fine_tuning_params(model):
    group_features = []
    group_hidden = []
    group_output   = []

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        if hasattr(model, 'features'):
            for name, param in model.named_parameters():
                if name.startswith("features"):
                    param.requires_grad = True
                    group_features.append(param)

        classifier_modules = list(model.classifier.children())
        num_cls_layers = len(classifier_modules)
        if num_cls_layers >= 1:
            last_module = classifier_modules[-1]
            output_param_names = set()
            for name, param in model.classifier.named_parameters():
                if isinstance(last_module, nn.Linear):
                    if (param.ndim == 2 and param.shape == last_module.weight.shape) or \
                       (param.ndim == 1 and param.shape == last_module.bias.shape):
                        output_param_names.add(name)
            for name, param in model.classifier.named_parameters():
                if name in output_param_names:
                    param.requires_grad = True
                    group_output.append(param)
                else:
                    param.requires_grad = True
                    group_hidden.append(param)
        else:
            for name, param in model.classifier.named_parameters():
                param.requires_grad = True
                group_output.append(param)

    elif hasattr(model, 'fc'):
        for name, param in model.named_parameters():
            if name.startswith("fc"):
                param.requires_grad = True
                group_output.append(param)
            
    elif hasattr(model, 'head'):
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True
                group_output.append(param)
            
    else:
        for param in model.parameters():
            param.requires_grad = True
            group_features.append(param)
            
    return group_features, group_hidden, group_output

