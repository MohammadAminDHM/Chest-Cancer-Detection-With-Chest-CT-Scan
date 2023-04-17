import json
import Core.utils as utils
import torch
import timm
import glob

def main():
    configs = json.load(open('./config.json', 'r'))

    # Model
    model = timm.create_model(configs['model']['ResNet50']['timm_name'], pretrained=True)

    model.head = torch.nn.Sequential(torch.nn.Linear(configs['Train']['ResNet50']['neuron'], 256),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, 10),
                                    torch.nn.Softmax()
                                    )

    model.load_state_dict(torch.load(configs['Test']['ResNet50']['model']))
    
    if torch.cuda.is_available():
        model = model.cuda()

    class_name = ['adenocarcinoma',
                'large.cell.carcinoma',
                'normal',
                'squamous.cell.carcinoma']
    
    true_count = 0
    all_data   = 0
    for class_file in class_name:
        image_path = glob.glob(configs['dataset']['Test'] + class_file + '/*')  
        all_data  += len(glob.glob(configs['dataset']['Test'] + class_file + '/*'))
        for image in image_path:    
            _ ,prediction= utils.app(image, model, 'cuda', class_name, False, False)
            print(class_file)
            if class_file == prediction:
                true_count += 1

        print('Accuracy of Model = {}'.format(true_count/all_data))

if __name__ == '__main__':
    main()