import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score, precision_score, recall_score


def logits_vs_true(logits,y_true,return_probs = False):
    '''Generates accuracy by using generated probabilities to compare against acceptance level and predict y.
    If return_probs is True then probabilties are returned along with accuracy.'''
    probabilities = torch.sigmoid(logits).detach() #applies sigmoid func giving probs for loss metric    
    predictions = (probabilities > 0.5).float()
    accuracy = (predictions == y_true).float().mean().item()
    if return_probs == True:
        return accuracy, probabilities
    return accuracy
    
def grad_norm(params):
    '''This calculates norm 2 of gradients calculate by the model to check the gradients are blowing up to extreme values.'''
    total = 0.0
    for param in params:
        if param.grad is not None:
                param_norm = param.grad.detach().norm(2)
                total += param_norm.item() ** 2
    total_norm = total ** 0.5 #Calculating norm 2 of norm2s 
    return total_norm
    
def positive_weight(y_train,device):
    '''Checks the training labels and from that determines proportion of positive vs negative.
    Allows altering of positive weight paramter in fit function'''
    positives = y_train.sum().item()
    negatives = len(y_train) - positives
    pos_weight = torch.tensor([negatives/max(positives,1)],device= device)
    return pos_weight

def threshold_free_metrics(probs,y_true):
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    roc = roc_auc_score(y_true, probs)
    pr  = average_precision_score(y_true, probs)
    return roc, pr

def select_threshold_constrained(
    probs, y_true, min_pos_rate=0.05,max_pos_rate=0.95,
    min_precision = None):
    '''Works to pick a specific F1 threshold under given constraints, passed as parameters
    Works only on validation data and retunrs dict of values'''
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    
    grid = np.linspace(0.05,0.95,181)
    best = {"t": 0.5, "f1": -1.0, "prec": 0.0, "rec": 0.0, "pos_rate": 0.0}
    for t in grid:
        yhat = (probs >= t) 
        pos_rate = float(yhat.mean())
        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue
        
        pr = precision_score (y_true,yhat,zero_division= 0)
        if (min_precision is not None) and (pr < min_precision):
            continue
        
        f1 = f1_score(y_true,yhat,zero_division= 0)
        rec = recall_score(y_true,yhat,zero_division=0)
        
        if f1>best['f1']:
            best = {"t": float(t), "f1": float(f1), "prec": float(pr), "rec": float(rec), "pos_rate": pos_rate}
        
    return best
        
def train_epoch(model,train_loader,optimizer,loss_metric,device,log_every=50):
    '''The code to train a single epoch of data,can be looped in juptyer notebook or running script.
    Returns training accuracy and loss for evaluation. '''
    n_trained = 0
    training_loss = 0.0
    training_accuracy = 0.0
    model.train()
    
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
def validate_training(model,val_loader,loss_metric,device,collect_probs = True):
    '''This validates the model on validation dataset to allow for tuning of hyperparameters.
    Returns val_accuracy and loss always and if collect_probs is True then also returns probabilities and labels.'''
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    num_evaluated = 0
    all_probs = []
    all_ys = []
    
    for i, (x_batch,y_batch) in enumerate(val_loader,1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(x_batch)
        loss= loss_metric(logits,y_batch)

        if collect_probs:
            accuracy,probs = logits_vs_true(logits,y_batch, return_probs= True)
            all_probs.append(probs.cpu())
            all_ys.append(y_batch.cpu())
        
        b_size = y_batch.size(0)
        num_evaluated += b_size
        val_loss += (loss.item() * b_size)
        val_accuracy += (accuracy * b_size)
        
    val_loss /= num_evaluated  
    val_accuracy /= num_evaluated
    
    if collect_probs:
        all_probs = torch.cat(all_probs).numpy().ravel()
        all_ys = torch.cat(all_ys).numpy().ravel().astype(float)
        return val_accuracy, val_loss, all_probs, all_ys
    
    return val_accuracy,val_loss

def fit(model,train_loader,val_loader,epochs,lr,device,save_path = 'models/best.pt',
        patience = None, pos_weight = None):
    '''This fit function automates the training of the model on multiple epochs also saves the best model parameters.
    Tests loss on each epoch and save the model with lowest loss.
    Returns model, t star score and history dict containing validation test scores.'''
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=1e-3)
    loss_metric = nn.BCEWithLogitsLoss(pos_weight= pos_weight) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor = 0.5, patience= 2, threshold=1e-3,
        cooldown= 0, min_lr= 1e-5
    )
    best_pr = -np.inf     
    best_epoch = 0
    best_t_star = 0.5
    wait = 0
    history = {"val_loss": [], "val_auc": [], "val_prauc": [], "t_star": []}
    
    for epoch in range(1,epochs+1):
        t_acc,t_loss = train_epoch(model,train_loader,optimizer,loss_metric,device,log_every=50)
        v_acc,v_loss,v_probs, v_ys = validate_training(model,val_loader,loss_metric,device,collect_probs= True)
        scheduler.step(v_loss)
        
        v_roc,v_pr = threshold_free_metrics(v_probs,v_ys)
    
        val_prev = float(np.mean(v_ys)) 

        tinfo = select_threshold_constrained(
            v_probs, v_ys,
            # Keep predicted positive rate near prevalenc
            min_pos_rate=max(0.05, val_prev - 0.05),
            max_pos_rate=min(0.95, val_prev + 0.05),
            # Don’t accept thresholds that have a low precision
            min_precision=max(0.55, val_prev),
        )

        if tinfo["f1"] < 0:
            # Pick threshold so pos_rate ~ prevalence 
            t_eq_prev = float(np.quantile(v_probs, 1.0 - val_prev))
            tinfo = {"t": t_eq_prev, "f1": -1.0, "prec": 0.0, "rec": 0.0, "pos_rate": val_prev}

        print(
            f"Epoch num: {epoch}"
            f"Train: loss {t_loss:.4f}, acc {t_acc:.4f}"
            f"Val: loss {v_loss:.4f}, acc {v_acc:.4f},ROC-AUC {v_roc:.3f},PR-AUC {v_pr:.3f}"
            f"Best Thr(F1) {tinfo['t']:.3f}, F1 {tinfo['f1']:.3f}, "
            f"P {tinfo['prec']:.3f}, R {tinfo['rec']:.3f}, PosRate {tinfo['pos_rate']:.3f}"
        )

        history["val_loss"].append(v_loss)
        history["val_auc"].append(v_roc)
        history["val_prauc"].append(v_pr)
        history["t_star"].append(tinfo["t"])
        
        if v_pr > best_pr + 1e-4:
            best_pr = v_pr
            best_epoch = epoch
            wait = 0
            best_t_star = tinfo["t"]
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New Model saved: PR-AUC = {v_pr:.4f} (t*={best_t_star:.3f})")
            
        else:
            wait += 1
            
        if patience is not None and wait >= patience:
            print(f'Ended on epoch {epoch}, due to patience.')
            break
        
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_t_star, history

@torch.no_grad()
def predict_probs(model, data_loader, device):
    '''Predicts probabilties for a given loader without calculating gradients or loss'''
    model.eval()
    all_probs , all_ys = [], []
    for xb,yb in data_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().numpy()
        all_probs.append(probs)
        all_ys.append(yb.numpy())
    probs = np.concatenate(all_probs, axis = 0)
    y_true = np.concatenate(all_ys,axis = 0)
    return probs,y_true

def thresholded_metrics(probs, y_true, t):
    """No threshold (ROC/PR) + thresholded (F1/Prec/Rec/PosRate) at t star val t."""
    probs  = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    # AUC guards (ROC needs both classes)
    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    pr  = average_precision_score(y_true, probs)

    yhat = (probs >= t)
    return {
        "roc": roc,
        "pr":  pr,
        "f1":  f1_score(y_true, yhat, zero_division=0),
        "prec": precision_score(y_true, yhat, zero_division=0),
        "rec":  recall_score(y_true, yhat, zero_division=0),
        "pos_rate": float(yhat.mean()),
        "t": float(t),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }

@torch.no_grad()
def evaluate_test(model, test_loader, device, t_star):
    """Oneshot test evaluation using frozen t* from validation step."""
    probs, y_true = predict_probs(model, test_loader, device)
    report = thresholded_metrics(probs, y_true, t_star)
    return report, probs, y_true

