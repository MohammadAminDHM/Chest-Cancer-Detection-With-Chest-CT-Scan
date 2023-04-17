import json
import Core.utils as utils
import torch
import timm
from torchvision.transforms import transforms
from torchvision import datasets
import json

def main():
    configs = json.load(open('./config.json', 'r'))

    #Prepare Dataset
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set_fire = configs['dataset']['Train']
    valid_set_fire = configs['dataset']['Valid']

    train_data = datasets.ImageFolder(train_set_fire, transform=transform)
    valid_data = datasets.ImageFolder(valid_set_fire, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=configs['Train']['batch'],
                                                    num_workers=configs['Train']['num_worker'],
                                                    shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=configs['Train']['batch'],
                                                    num_workers=configs['Train']['num_worker'],
                                                    shuffle=True)

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    model = timm.create_model(configs['Train']['path']['timm_name'], pretrained=True)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    model.head = torch.nn.Sequential(torch.nn.Linear(configs['Train']['ResNet50']['neuron'], 256),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, 2),
                                    torch.nn.Softmax()
                                    )

    for param in model.head.parameters():
        param.requires_grad = True

    if use_cuda:
        model_transfer = model.cuda()
    else:
        model_transfer = model

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_transfer.head.parameters(), lr=configs['Train']['lr'], weight_decay=configs['Train']['wd'])

    model = utils.train(configs['Train']['epoch'],
                        loaders,
                        model,
                        optimizer,
                        criterion,
                        True,
                        configs['Train']['path']['model_path'],
                        configs['Train']['path']['result_path'],
                        configs['Train']['ResNet50']['model_name'])


