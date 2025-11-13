import pandas as pd

df = pd.read_csv('bert_wrong_cls_best_model.csv')

count_pred_0_GT_1 = 0
count_pred_0_GT_2 = 0
count_pred_1_GT_2 = 0
count_pred_1_GT_0 = 0
count_pred_2_GT_0 = 0
count_pred_2_GT_1 = 0

for index, row in df.iterrows():
    if row['all_preds_cls'] == 0 and row['all_labels_cls'] == 1:
        count_pred_0_GT_1 += 1
    elif row['all_preds_cls'] == 0 and row['all_labels_cls'] == 2:
        count_pred_0_GT_2 += 1
    elif row['all_preds_cls'] == 1 and row['all_labels_cls'] == 0:
        count_pred_1_GT_0 += 1
    elif row['all_preds_cls'] == 1 and row['all_labels_cls'] == 2:
        count_pred_1_GT_2 += 1
    elif row['all_preds_cls'] == 2 and row['all_labels_cls'] == 0:
        count_pred_2_GT_0 += 1   
    elif row['all_preds_cls'] == 2 and row['all_labels_cls'] == 1:
        count_pred_2_GT_1 += 1    
        
print(count_pred_0_GT_1) # 279
print(count_pred_0_GT_2) # 59
print(count_pred_1_GT_0) # 433
print(count_pred_1_GT_2) # 27 
print(count_pred_2_GT_0) # 180
print(count_pred_2_GT_1) # 5
    