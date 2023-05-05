
import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Create dummy data with class imbalance 99 to 1
class_counts = torch.tensor([104, 642, 784])
numDataPoints = class_counts.sum()
data_dim = 5
bs = 170
data = torch.randn(numDataPoints, data_dim)

target = torch.cat((torch.zeros(class_counts[0], dtype=torch.long),
                    torch.ones(class_counts[1], dtype=torch.long),
                    torch.ones(class_counts[2], dtype=torch.long) * 2))

print('target train 0/1/2: {}/{}/{}'.format(
    (target == 0).sum(), (target == 1).sum(), (target == 2).sum()))

# Compute samples weight (each sample should get its own weight)
class_sample_count = torch.tensor(
    [(target == t).sum() for t in torch.unique(target, sorted=True)])
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in target])

# Create sampler, dataset, loader
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataset = torch.utils.data.TensorDataset(data, target)
#train_dataset = triaxial_dataset(data, target)
train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=0, sampler=sampler)

# Iterate DataLoader and check class balance for each batch
for i, (x, y) in enumerate(train_loader):
    print("batch index {}, 0/1/2: {}/{}/{}".format(
        i, (y == 0).sum(), (y == 1).sum(), (y == 2).sum()))
def one_input_training_loop(n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for data1, labels in tqdm(train_loader, desc = "batch", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data1 = data1.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(data1)
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = one_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                                
def two_input_training_loop(n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    early_stopper = EarlyStopper(patience=8, min_delta=0.0001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for data1, data2, labels in tqdm(train_loader, desc = "batch", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(data1, data2)
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = two_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                               
def three_input_training_loop(n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    early_stopper = EarlyStopper(patience=8, min_delta=0.0001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for data1, data2, data3, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            data3 = data3.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(data1, data2)
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = three_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch


def two_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for data1, data2, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(data1, data2)
            #probs = torch.nn.Softmax(outputs)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        pred_cpu = torch.cat(predictions).cpu()
        labels_cpu =  torch.cat(all_labels).cpu()
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            m = torch.nn.Softmax(dim=1)
            pred_cpu = torch.cat(predictions).cpu()
            labels_cpu =  torch.cat(all_labels).cpu()
            torch.save(
                {   'predict_proba' : m(torch.cat(predict_proba)),
                    'predictions' : pred_cpu.numpy(),
                    'labels_val' : labels_cpu.numpy(),
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val,
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss,  f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro')

def three_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for data1, data2, data3, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            data3 = data3.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(data1, data2, data3)
            #probs = torch.nn.Softmax(outputs)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        pred_cpu = torch.cat(predictions).cpu()
        labels_cpu =  torch.cat(all_labels).cpu()
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            m = torch.nn.Softmax(dim=1)
            pred_cpu = torch.cat(predictions).cpu()
            labels_cpu =  torch.cat(all_labels).cpu()
            torch.save(
                {   'predict_proba' : m(torch.cat(predict_proba)),
                    'predictions' : pred_cpu.numpy(),
                    'labels_val' : labels_cpu.numpy(),
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val,
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss,  f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro')

# ----------------------------------------- Test Loops ----------------------------------------- #

def two_input_test_loop(model, loss_fn, loss_fn_str, test_loader, device):
    '''
    Assessing trained model on test dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    test_loader: generator creating batches of test data
    '''
    model = model.to(device)
    model.eval()
    loss_test = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on test data.
        for data1, data2, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = False):
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            labels = labels.to(device= device)

            # Assessing outputs
            outputs = model(data1, data2)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_test += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == torch.max(labels, 1)[1]).sum())
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + torch.max(labels, 1)[1].tolist()
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss, all_predictions, all_labels

def three_input_test_loop(model, loss_fn, loss_fn_str, test_loader, device):
    '''
    Assessing trained model on test dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    test_loader: generator creating batches of test data
    '''
    model = model.to(device)
    model.eval()
    loss_test = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on test data.
        for data1, data2, data3, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = False):
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            data3 = data3.to(device = device)
            labels = labels.to(device= device)

            # Assessing outputs
            outputs = model(data1, data2, data3)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_test += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == torch.max(labels, 1)[1]).sum())
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + torch.max(labels, 1)[1].tolist()
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss, all_predictions, all_labels


#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#
#                                Feature Extraction Loops
#---------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#


def two_input_training_loop_fe(n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for data1, data2, labels in tqdm(train_loader, desc = "batch", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(data1, data2)
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = two_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
  

def three_input_training_loop_fe(n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for data1, data2, data3, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            data3 = data3.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(data1, data2)
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Stopping training.")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = three_input_validation_loop(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device, model_name)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                                
 # ----------------------------- Feature Extraction Validation Loop -----------------------------# 
                              

def two_input_validation_loop_fe(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    model = model.to(device)
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for data1, data2, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(data1, data2)
            #probs = torch.nn.Softmax(outputs)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
    model.train()
    return correct, total, avg_val_loss, best_val_loss
    

def three_input_validation_loop_fe(model, loss_fn, loss_fn_str, valid_loader, best_val_loss, device):
    '''
   
    '''
    model = model.to(device)
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for data1, data2, data3, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            data1 = data1.to(device = device)
            data2 = data2.to(device = device)
            data3 = data3.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(data1, data2, data3)
            #probs = torch.nn.Softmax(outputs)
            if loss_fn_str == 'BCE' or loss_fn_str == 'focal':
                loss = loss_fn(outputs,labels)
            else:
                loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
    model.train()
    return correct, total, avg_val_loss, best_val_loss
