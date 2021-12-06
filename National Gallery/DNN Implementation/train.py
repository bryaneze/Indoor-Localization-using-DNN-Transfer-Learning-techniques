import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import haversine
import model
import argparse
from model import Classifier
from dataset import trainloader, testloader
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
import statistics
import os 

matplotlib.style.use('ggplot')


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

print("here")


lr = 0.003 
epochs = 250
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

model = model.Classifier()
model.to(device)
#optimizer = optim.Adam(model.parameters(), lr =lr)
#optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.09)
optimizer = optim.SGD(model.parameters(),lr=lr)
criterion = nn.MSELoss()

# strings to save the loss plot, accuracy plot, and model with different ...
# ... names according to the training type
# if not using `--lr-scheduler` or `--early-stopping`, then use simple names
loss_plot_name = 'n_loss'
acc_plot_name = 'n_dist'
model_name = 'model'

# either initialize early stopping or learning rate scheduler
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_dist'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_dist'
    model_name = 'es_model'

# training function
def fit(model, train_dataloader, optimizer, criterion):
    train_losses = []
    pred_train_list, train_label_list, train_dist = [],[],[]
   # print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    #prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for data, target in train_dataloader:
        tot_train_loss = 0
        counter += 1
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0],-1)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data.float())
        loss = torch.sqrt(criterion(outputs, target.float()))
        tot_train_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
      #  train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = tot_train_loss / len(trainloader.dataset)
    train_losses.append(train_loss)
    #print(len(outputs))
    for i in range(len(outputs)):
        pred_train_list.append(outputs[i])
        train_label_list.append(target[i])
    for j in range(len(pred_train_list)):
        try:
            train_dist.append(haversine.haversine(pred_train_list[j][1].float(),pred_train_list[j][0].float(),train_label_list[j][1].float(),train_label_list[j][0].float()))
        except ValueError:
            train_dist.append(0)
   
    train_accuracy = statistics.median(train_dist)
    #print(train_accuracy)
    #print('here2')
    return train_loss, train_accuracy



# validation function
def validate(model, test_dataloader, criterion):
    test_losses = []
    pred_test_list, test_label_list, test_dist = [],[],[]
   # print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
   # prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for data, target in test_dataloader:
            tot_test_loss = 0 
            counter += 1
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0],-1)
            total += target.size(0)
            outputs = model(data.float())
            loss = torch.sqrt(criterion(outputs, target.float()))
            tot_test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
           # val_running_correct += (preds == target).sum().item()
        
        test_loss = tot_test_loss / len(testloader.dataset)
        test_losses.append(test_loss)
        for i in range(len(outputs)):
            pred_test_list.append(outputs[i])
            test_label_list.append(target[i])

        for j in range(len(pred_test_list)):
            try:
                test_dist.append(haversine.haversine(pred_test_list[j][1].float(),pred_test_list[j][0].float(),test_label_list[j][1].float(),test_label_list[j][0].float()))
            except ValueError:
                test_dist.append(0)


        val_accuracy = statistics.median(test_dist)
        return test_loss, val_accuracy



# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(
        model, trainloader, optimizer, criterion
    )
    val_epoch_loss, val_epoch_accuracy = validate(
        model, testloader, criterion
    )
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    print(f"Train Loss: {train_epoch_loss:.4f}, Median train diff distance: {train_epoch_accuracy:.2f}m")
    print(f'Val Loss: {val_epoch_loss:.4f}, Median val diff distance: {val_epoch_accuracy:.2f}m')

end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")



print('Saving loss and accuracy plots...')
# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train distance')
plt.plot(val_accuracy, color='blue', label='validataion distance')
plt.xlabel('Epochs')
plt.ylabel('Distance (m)')
plt.legend()
plt.savefig(f"./outputs/{acc_plot_name}.png")
plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"./outputs/{loss_plot_name}.png")
plt.show()
    
# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), f"./outputs/{model_name}.pth")
 
print('TRAINING COMPLETE')


# train_losses, test_losses = [],[]
#for e in range(epochs):
#     predicted_list, label_list,dist = [],[],[]
#     tot_train_loss = 0
#     for data, labels in trainloader:
#         data,labels = data.to(device), labels.to(device)
#         data = data.view(data.shape[0],-1)
#         data = data.float()
#         optimizer.zero_grad()
#         output = model(data)
#         #output = ((output - torch.mean(output))/torch.max(output)-torch.min(output))
#         loss = torch.sqrt(criterion(output,labels.float())) #RMSE
#         #loss = criterion(output,labels.float())
#         tot_train_loss+=loss.item()
# 
#         loss.backward()
#         optimizer.step()
# 
#     else:
#         model.eval()
#         tot_test_loss = 0
#         with torch.no_grad():
# 
#             for data, labels in testloader:
#                 data,labels = data.to(device), labels.to(device)
#                 data = data.view(data.shape[0],-1)
#                 data=data.float()
#                 test_output = model(data)
#                 loss = torch.sqrt(criterion(test_output,labels.float()))
#                 #loss = criterion(test_output,labels.float())
#                 tot_test_loss+=loss.item()
#         model.train()
# 
#         train_loss = tot_train_loss/len(trainloader.dataset)
#         test_loss = tot_test_loss/len(trainloader.dataset)
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
# 
#         #Add predicted and label value into 2 lists for distance calculation
#         for i in range(len(test_output)):
#             predicted_list.append(test_output[i])
#             label_list.append(labels[i])
#         #print(predicted_list)
#         #print("new")
# 
#         for j in range(len(predicted_list)):
# 
#             try:
#                 dist.append(haversine.haversine(predicted_list[j][1].float(),predicted_list[j][0].float(),label_list[j][1].float(),label_list[j][0].float()))
#             except ValueError:
#                 dist.append(0)
#         try:
#             print("Epoch:{}..".format(e+1),
#                  #"Training Loss:{:.3f}..".format(train_loss),
#                  "Test Loss:{:.3f}..".format(test_loss),
#                   "Min Dist Diff:{:.3f}..".format(min(dist)),
#                   "Median Dist Diff:{:.3f}..".format(statistics.median(dist)),
#                  "Avg Dist Diff:{:.3f}..".format(sum(dist)/len(dist)))
#         except ValueError:
#             print("error")
# 
