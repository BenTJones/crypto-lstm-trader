import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

def logits_vs_true(logits,y_true):
    probabilities = torch.sigmoid(logits) #applies sigmoid func giving probs for loss metric    
    predictions = (probabilities > 0.5).float()
    correct_frac = (predictions == y_true).float().mean().item()
    return correct_frac
    
def grad_norm(params):
        total = 0.0
        for param in params:
            if param.grad is not None:
                param_norm = param.grad.detach().norm(2)
                total += param_norm.item() ** 2
        total_norm = total ** 0.5 #Calculating norm 2 of norm2s 
        return total_norm
    
def train_epoch(model,train_loader,optimizer,loss_metric,device,log_every=50):
    n_trained = 0
    training_loss = 0.0
    training_accuracy = 0.0
    
    for i, (x_batch,y_batch) in enumerate(train_loader,1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        #sets up the optimzer resetting prev grads and moving items to device
        
        logits = model(x_batch)
        loss= loss_metric(logits,y_batch)
        loss.backward()
        
        prev_grad_norm = grad_norm(model.parameters())
        
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
        #This block contains forward pass, loss calc, back propogation,
        # stops excessive gradients and updates weights
        #BackProp is the same maths and principles as PET,MRI from MMM
        
        with torch.no_grad():
            accuracy = logits_vs_true(logits,y_batch)
            b_size = y_batch.size(0)
            n_trained += b_size
            training_loss += (loss.item() * b_size)
            training_accuracy += (accuracy * b_size)
        
            if i % log_every == 0 or i == 1:  
                print(f'Training results of batch {i}'
                    f'Loss : {loss:4f}, Accuracy : {accuracy:4f}'
                    f'Pre-Clip param norm: {prev_grad_norm:4f}')
                
    training_loss /= n_trained
    training_accuracy /= n_trained
    return training_accuracy, training_loss
   
@torch.no_grad() #This function doesn't need to calc gradients so can save memory and speed up eval
def validate_training(model,val_loader,loss_metric,device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    num_evaluated = 0
    
    for i, (x_batch,y_batch) in enumerate(val_loader,1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(x_batch)
        loss= loss_metric(logits,y_batch)

        accuracy = logits_vs_true(logits,y_batch)
        b_size = y_batch.size(0)
        num_evaluated += b_size
        val_loss += (loss.item() * b_size)
        val_accuracy += (accuracy * b_size)
        
    val_loss /= num_evaluated  
    val_accuracy /= num_evaluated
    
    return val_accuracy,val_loss

def fit(model,train_loader,val_loader,epochs,lr,device,save_path = 'models/best.pt', patience = None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_metric = nn.BCEWithLogitsLoss() 
    
    best_val_loss = float('inf')
    best_epoch = 0
    wait = 0 #For patience implementation
    
    for epoch in range(1,epochs+1):
        t_acc,t_loss = train_epoch(model,train_loader,optimizer,loss_metric,device,log_every=100)
        v_acc,v_loss = validate_training(model,val_loader,loss_metric,device)
        
        print(f'Epoch {epoch}'
              f'Training: Loss {t_loss:.4f},Accuracy {t_acc:4f}'
              f'Evaluation: Loss {v_loss:4f},Accuracy {v_acc:4f}')
        
        if best_val_loss > v_loss:
            best_epoch = epoch
            wait = 0 
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f'New Model saved: Loss = {v_loss:.4f} ')
            
        else:
            wait += 1
            
        if patience is not None and wait >= patience:
            print(f'Ended on epoch {epoch}, due to patience.')
        
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model