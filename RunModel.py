# -*- coding:utf-8 -*-

import os
import random
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import SCDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader
from utils.ContrastiveLoss import GraphContrastiveLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        num_classes = pred.size(1)
        target_one_hot = torch.zeros_like(pred).scatter(1, target.long().unsqueeze(1), 1)
        target_smooth = target_one_hot * (1.0 - self.smoothing) + (self.smoothing / num_classes)
        loss = self.bce(pred, target_smooth)
        return loss

def run_SC_model(SEED, DATASET, MODEL, K_Fold, LOSS, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    hp = hyperparameter()

    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    if DATASET == "BD2D":
        split_pos = 52010
        train_data_list = data_list[:split_pos]
        test_data_list = data_list[split_pos:]
        print("data shuffle")
        train_data_list = shuffle_dataset(train_data_list, SEED)
    else:
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
    
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        train_dataset = ProteinMoleculeDataset(train_dataset, ligand_dict, protein_dict, device=device)
        valid_dataset = ProteinMoleculeDataset(valid_dataset, ligand_dict, protein_dict, device=device)
        test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)

        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
                                        
        model = MODEL(device=device)

        bert_ids = list(map(id, model.drug_encoder.parameters()))
        for name, p in model.named_parameters():
            if id(p) not in bert_ids and p.dim() > 1:
                nn.init.xavier_uniform_(p)

        bert_params_id = list(map(id, model.drug_encoder.parameters()))
        base_params = list(filter(lambda p: id(p) not in bert_params_id, model.parameters()))
        
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': hp.Learning_rate},
            {'params': model.drug_encoder.parameters(), 'lr': hp.Learning_rate * 0.1} 
        ], weight_decay=hp.weight_decay) 
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=hp.Epoch, 
            eta_min=1e-6
        )

        Loss = LabelSmoothingLoss(smoothing=0.1).to(device)
        CL_Loss = GraphContrastiveLoss(temperature=0.1).to(device)
        alpha = 1.0

        save_path = "./" + DATASET + "/SC-DTI"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)
        
        scaler = torch.amp.GradScaler('cuda')
        training_history = [] 

        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break

            train_losses_in_epoch = []
            model.train()
            
            accumulation_steps = 4
            
            for i, data in enumerate(train_loader):
                data = data.to(device)
                
                with torch.amp.autocast('cuda'):
                    predicted_y, drug_feat, prot_feat = model(data, return_features=True)
                    
                    drug_feat = F.normalize(drug_feat, p=2, dim=1)
                    prot_feat = F.normalize(prot_feat, p=2, dim=1)
                    
                    train_loss_cls = Loss(predicted_y, data.cls_y)
                    train_loss_cl = CL_Loss(drug_feat, prot_feat, data.cls_y.float())
                    
                    train_loss = (train_loss_cls + alpha * train_loss_cl) / accumulation_steps

                scaler.scale(train_loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    
                    torch.cuda.empty_cache()

                train_losses_in_epoch.append(train_loss.item() * accumulation_steps)
                
            scheduler.step()
            
            train_loss_a_epoch = np.average(train_losses_in_epoch)

            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for data in valid_loader:
                    data = data.to(device)
                    valid_scores = model(data)
                    valid_labels = data.cls_y
                    
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    
                    if valid_scores.dim() == 1 or (valid_scores.dim() == 2 and valid_scores.shape[1] == 1):
                        valid_scores = torch.sigmoid(valid_scores).to('cpu').data.numpy()
                        if valid_scores.ndim == 2: 
                            valid_scores = valid_scores.squeeze(1)
                        valid_predictions = (valid_scores > 0.5).astype(int)
                    else:
                        valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                        valid_predictions = np.argmax(valid_scores, axis=1)
                        valid_scores = valid_scores[:, 1] 

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)
            
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss_a_epoch,
                'valid_loss': valid_loss_a_epoch,
                'valid_auc': AUC_dev,
                'valid_aupr': PRC_dev
            })

            early_stopping(Accuracy_dev, model, epoch)
            history_df = pd.DataFrame(training_history)
            history_df.to_csv(save_path + '/training_history.csv', index=False)
            print(f"Training history saved to {save_path}/training_history.csv")

        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{device}.pth', weights_only=True))

        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, SC=True)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, SC=True)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, SC=True)
            
        extract_length_analysis_data(model, test_loader, save_path, device, model_name="SC-DTI")
        
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')
        break

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)

def ensemble_run_SC_model(SEED, DATASET, K_Fold, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    hp = hyperparameter()

    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()  

    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(SCDTI().to(device))
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device))) 
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'SCDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = LabelSmoothingLoss(smoothing=0.1).to(device)

    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, SC=True)
    
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
    
def extract_length_analysis_data(model, test_loader, save_path, device, model_name="SC-DTI"):
    print(f"Extracting length analysis data for {model_name}...")
    model.eval()
    records = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            valid_scores = model(data)
            valid_labels = data.cls_y
            
            if valid_scores.dim() == 1 or (valid_scores.dim() == 2 and valid_scores.shape[1] == 1):
                valid_scores = torch.sigmoid(valid_scores).cpu().data.numpy()
                if valid_scores.ndim == 2: 
                    valid_scores = valid_scores.squeeze(1)
            else:
                valid_scores = F.softmax(valid_scores, 1).cpu().data.numpy()
                valid_scores = valid_scores[:, 1] 

            valid_labels = valid_labels.cpu().data.numpy()
            
            try:
                lengths = torch.bincount(data.prot_node_aa_batch).cpu().numpy()
            except Exception as e:
                print("Warning: Cannot extract lengths. Make sure follow_batch=['prot_node_aa'] is used.")
                lengths = [0] * len(valid_labels)
                
            for i in range(len(valid_labels)):
                if valid_labels[i] == 1:
                    records.append({
                        'Protein_Length': lengths[i],
                        'Confidence': valid_scores[i],
                        'Model': model_name
                    })
                    
    df = pd.DataFrame(records)
    out_file = f"{save_path}/{model_name}_length_analysis.csv"
    df.to_csv(out_file, index=False)
    print(f"Length analysis data saved to {out_file}")