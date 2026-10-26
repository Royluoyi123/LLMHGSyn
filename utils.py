import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,auc, roc_curve, average_precision_score,cohen_kappa_score, f1_score, accuracy_score,precision_recall_curve,balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef,recall_score,confusion_matrix
from scipy.stats import pearsonr,spearmanr
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import random



def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score,roc_curve,cohen_kappa_score,precision_score,recall_score,mean_absolute_error,balanced_accuracy_score,matthews_corrcoef
from scipy.stats import pearsonr,spearmanr
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import random


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def get_roc_score(labels, preds):

    auc_score = AUC(labels, preds)
    aupr_score = AUPR(labels, preds)
    acc_score = ACC(labels, preds)
    kappa_score = KAPPA(labels, preds)
    bacc_score = BACC(labels, preds)
    f1_score = F1(labels, preds)
    mcc_score = MCC(labels, preds)
    recall_score_value = RECALL(labels, preds)
    spec_score = SPEC(labels, preds)

    return auc_score, aupr_score, acc_score, kappa_score, bacc_score, f1_score,mcc_score,recall_score_value,spec_score


def get_mse_score(preds, labels):

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson_corr, pearson_p = pearsonr(labels, preds)
    spearman_corr, spearman_p = spearmanr(labels, preds)

    return mse, rmse, pearson_corr, r2 ,mae, spearman_corr

def AUC(ground_truth, prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def AUPR(ground_truth,prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    precision,recall,thresholds=precision_recall_curve(ground_truth, prediction)
    res=auc(recall,precision)
    #res = average_precision_score(y_true=ground_truth, y_score=prediction)
    return(res)

def ACC(ground_truth,prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    #fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    
    #youden_thresh = round(youden_thresh, 3)
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]
    return accuracy_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def get_best_threshold_by_f1(ground_truth, prediction):
    ground_truth = np.array(ground_truth).astype(int)
    prediction = np.array(prediction).reshape(-1)

    precision, recall, thresholds = precision_recall_curve(
        ground_truth,
        prediction
    )

    # 注意：precision 和 recall 的长度比 thresholds 多 1
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold

def MCC(ground_truth, prediction):
    best_threshold = get_best_threshold_by_f1(ground_truth, prediction)

    y_true = np.array(ground_truth).astype(int)
    y_pred = (np.array(prediction) >= best_threshold).astype(int)

    return matthews_corrcoef(y_true, y_pred)

def RECALL(ground_truth, prediction):
    best_threshold = get_best_threshold_by_f1(ground_truth, prediction)

    y_true = np.array(ground_truth).astype(int)
    y_pred = (np.array(prediction) >= best_threshold).astype(int)

    return recall_score(y_true, y_pred, zero_division=0)

def SPEC(ground_truth, prediction):
    best_threshold = get_best_threshold_by_f1(ground_truth, prediction)

    y_true = np.array(ground_truth).astype(int)
    y_pred = (np.array(prediction) >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return specificity

def KAPPA(ground_truth,prediction):
    #fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    
    #youden_thresh = round(youden_thresh, 3)

    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]
    
    return cohen_kappa_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def BACC(ground_truth,prediction):
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]

    return balanced_accuracy_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def F1(ground_truth, prediction):
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)
    
    # 计算每个阈值下的F1-score（加1e-10防止除零）
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 返回最大的F1值
    return f1_scores.max()

def eval_threshold(labels_all, preds_all):

    # fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = 0.5
    preds_all_ = []
    
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)

    return preds_all, preds_all_


class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df
    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """

    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)


def get_cfps(mol, radius=2, nBits=256, useFeatures=False, counts=False, dtype=np.float32):
    """Calculates circural (Morgan) fingerprint.
    http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn
    Returns
    -------
    ML.FP
        Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures,
                                                   bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))


def get_fingerprint_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp


def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




