from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import cv2
import time
import json

def cma(cm):
  """Calculate classification criterias

  Args:
      cm (np): Confusion Matrix

  Return:
      list: [mean recall of classes, mean precision of classes, mean F1 of classes]
  """
  
  recall    = []
  precision = []
  f1        = []
  for i in cm.shape[0]:
    recall.append(cm[i][i]/cm.sum(axis=0)[i])
    precision.append(cm[i][i]/cm.sum(axis=1)[i])
    f1.append(2 * (((cm[i][i]/cm.sum(axis=0)[i]) * (cm[i][i]/cm.sum(axis=1)[i]))/((cm[i][i]/cm.sum(axis=0)[i]) + (cm[i][i]/cm.sum(axis=1)[i]))))

  recall    = [x for x in recall if str(x) != 'nan']
  precision = [x for x in precision if str(x) != 'nan']
  f1        = [x for x in f1 if str(x) != 'nan']

  return [sum(recall)/len(recall), sum(precision)/len(precision), sum(f1)/len(f1)]

def train(n_epochs, loader, model, optimizer, criterion, use_cuda, save_model, save_result,  model_name):
    """Train model

  Args:
      n_epochs (int): Count of epochs
      loader : Dataset(Train & Valid)
      model : Model
      optimizer : Optimizer
      criterion : Cost Function
      use_cuda(bool) :True(Use CUDA), False(Use CPU) 
      save_model(str) : Directory of save models
      save_result(str) : Directory of save result
      model_name (str): Model's name

  Return:
      Trained model
    """    
    train_accuracy_list = []
    train_loss_list = []

    train_recall_list = []
    train_precision_list = []
    train_f1_list = []

    valid_accuracy_list = []
    valid_loss_list = []

    valid_recall_list = []
    valid_precision_list = []
    valid_f1_list = []

    valid_acc_max = 0.0
    valid_r_max = 0.0
    valid_p_max = 0.0
    valid_f1_max = 0.0    

    train_Confusion_matrix = np.zeros((4, 4)) 
    valid_Confusion_matrix = np.zeros((4, 4))        

    for epoch in range(1, (n_epochs + 1)):
        
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        
        model.train()
        start = time.time()
        for batch_idx, (data, target) in enumerate(loader['train']):
            target_1 = target
            if use_cuda:
                data, target_2 = data.cuda(), target.cuda()            

            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target_2)
            loss.backward()
            optimizer.step()
            train_Confusion_matrix += cm(preds.cpu().detach().numpy(), target_1.numpy(), labels=[i for i in range(4)])                      
            train_acc = train_acc + torch.sum(preds == target_2.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                
                
        model.eval()
        for batch_idx, (data, target) in enumerate(loader['valid']):
            target_1 = target
            if use_cuda:
                data, target_2 = data.cuda(), target.cuda()                
            output = model(data)
            
            _, preds = torch.max(output, 1)
            loss = criterion(output, target_2)            
            
            valid_Confusion_matrix += cm(preds.cpu().detach().numpy(), target_1, labels=[i for i in range(4)])
            valid_acc = valid_acc + torch.sum(preds == target_2.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        train_loss = train_loss/len(loader['train'].dataset)
        valid_loss = valid_loss/len(loader['valid'].dataset)
        train_acc = train_acc/len(loader['train'].dataset)
        valid_acc = valid_acc/len(loader['valid'].dataset)
        
        train_result = cma(train_Confusion_matrix)
        valid_result = cma(valid_Confusion_matrix)

        train_recall_list.append(train_result[0])
        train_precision_list.append(train_result[1])
        train_f1_list.append(train_result[2])

        valid_recall_list.append(valid_result[0])
        valid_precision_list.append(valid_result[1])
        valid_f1_list.append(valid_result[2])
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)
        
        Epoch_time = time.time() - start
        print('Epoch: {}\
              \n Epoch Time: {:6f}\
              \nTraining Acc: {:6f}\
              \nTraining Loss: {:6f}\
              \nTraining recall: {:6f}\
              \nTraining precision: {:6f}\
              \nTraining F1 Score: {:6f}\
              \n\n\
              \nValidation Acc: {:6f}\
              \nValidation Loss: {:.6f}\
              \nValidation recall: {:6f}\
              \nValidation precision: {:6f}\
              \nValidation F1 Score: {:6f}           '.format(
                                                        epoch,
                                                        Epoch_time,
                                                        train_acc,
                                                        train_loss,
                                                        train_result[0],
                                                        train_result[1],
                                                        train_result[2],
                                                        valid_acc,
                                                        valid_loss,
                                                        valid_result[0],
                                                        valid_result[1],
                                                        valid_result[2]
                                                        ))

        if valid_acc >= valid_acc_max:
            print('\n\nValidation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_acc_max,
            valid_acc))
            torch.save(model.state_dict(), save_model + 'model_' + model_name + '_acc_max.pt')
            valid_acc_max = valid_acc

        if valid_result[0] >= valid_r_max:
            print('\n\nValidation recall true increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_r_max,
            valid_result[0]))
            torch.save(model.state_dict(), save_model + 'model_' + model_name + '_r_max.pt')
            valid_r_max = valid_result[0]  

        if valid_result[1] >= valid_p_max:
            print('\n\nValidation precision true increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_p_max,
            valid_result[1]))
            torch.save(model.state_dict(), save_model + 'model_' + model_name + '_p_max.pt')
            valid_p_max = valid_result[1]
            
        if valid_result[2] >= valid_f1_max:
            print('\n\nValidation F1 true increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_f1_max,
            valid_result[2]))
            torch.save(model.state_dict(), save_model + 'model_' + model_name + '_f1_max.pt')
            valid_f1_max = valid_result[2]


        print('\n\n ------------------------------------------------------- \n\n')  
    
    
    train_loss_list     = [i.cpu().tolist() for i in train_loss_list]
    valid_loss_list     = [i.cpu().tolist() for i in valid_loss_list]
    train_accuracy_list = [i.cpu().tolist() for i in train_accuracy_list]
    valid_accuracy_list = [i.cpu().tolist() for i in valid_accuracy_list]

    with open(save_result + 'model_' + model_name + '_train_accuracy_list.json', 'w') as f:
      json.dump(train_accuracy_list, f, indent=2)

    with open(save_result + 'model_' + model_name + '_train_loss_list.json', 'w') as f:
      json.dump(train_loss_list, f, indent=2)

    with open(save_result + 'model_' + model_name + '_train_recall_list.json', 'w') as f:
      json.dump(train_recall_list, f, indent=2)      

    with open(save_result + 'model_' + model_name + '_train_precision_list.json', 'w') as f:
      json.dump(train_precision_list, f, indent=2)
                  
    with open(save_result + 'model_' + model_name + '_train_f1_list.json', 'w') as f:
      json.dump(train_f1_list, f, indent=2)
            
    with open(save_result + 'model_' + model_name + '_valid_accuracy_list.json', 'w') as f:
      json.dump(valid_accuracy_list, f, indent=2)

    with open(save_result + 'model_' + model_name + '_valid_loss_list.json', 'w') as f:
      json.dump(valid_loss_list, f, indent=2)

    with open(save_result + 'model_' + model_name + '_valid_recall_list.json', 'w') as f:
      json.dump(valid_recall_list, f, indent=2)      

    with open(save_result + 'model_' + model_name + '_valid_precision_list.json', 'w') as f:
      json.dump(valid_precision_list, f, indent=2)
                  
    with open(save_result + 'model_' + model_name + '_valid_f1_list.json', 'w') as f:
      json.dump(valid_f1_list, f, indent=2)
        
    return model

def predict(image, model, device, class_name):
    """Predict result with our model

  Args:
      image : input image
      model : Trained model
      device(str) : 'cuda' or 'cpu'
      class_name(list) : list of name classes

  Returns:
      prob : measure of detection(percent)
      class_name[idx] : name of output class
    """
    
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(), 
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
      image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    except:
      image = image.convert('RGB')
      image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    
    if device == 'cuda':
        if torch.cuda.is_available():
            image = image.cuda()
        else:
            print("You don't have cuda")

    with torch.no_grad():      
      model.eval()
      pred = model(image)

      
    idx = torch.argmax(pred)

    prob = pred[0][idx].item()*100
    
    return prob, class_name[idx]
  
def app(path, model, device, class_name, display_prob, show):
    """Use model

  Args:
      path (str): Path of input image
      model : Our model
      device(str) : 'cuda' or 'cpu'
      class_name(list) : list of name classes
      display_prob (bool): True(show measure of detection(percent)) False(Doesn't show)
      show (bool) : True(show image) False(Doesn't show)
  Returns:
      prob : measure of detection(percent)
      result : name of output class
  """      
    img = Image.open(path)
    if show:
      plt.imshow(img)
      plt.show()

    prob, result = predict(img, model, device, class_name)
    if display_prob:
      print('Probability of {} : {:.6f}'.format(result, prob))
                  
    return prob, result
