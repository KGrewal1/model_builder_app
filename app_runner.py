"""
For building of ChemProp based models
"""
#import packages
from multiprocessing import Pool
import os
import shutil
from typing import List, Tuple



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import chemprop
from chemprop.data import get_smiles
from chemprop.features import get_features_generator, load_features, save_features
from chemprop.utils import makedirs

#for featurization

def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.
    Assumes temporary files are named in order 0.npz, 1.npz, ...
    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def generate_and_save_features(data_path: str, save_path: str, smiles_column: str=None, features_generator: str = 'rdkit_2d_normalized', save_frequency: int = 10000, restart: bool = True, sequential: bool = False ):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.
    :param args: Arguments.
    """
    # Create directory for save_path
    makedirs(save_path, isfile=True)

    # Get data and features function
    smiles = get_smiles(path=data_path, smiles_columns=smiles_column, flatten=True)
    features_generator = get_features_generator(features_generator)
    temp_save_dir = save_path + '_temp'

    # Load partially complete data
    if restart:
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(save_path):
            raise ValueError(f'"{save_path}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
        features, temp_num = [], 0

    # Build features map function
    smiles = smiles[len(features):]  # restrict to data for which features have not been computed yet

    if sequential:
        features_map = map(features_generator, smiles)
    else:
        features_map = Pool().imap(features_generator, smiles)

    # Get features
    temp_features = []
    for i, feats in tqdm(enumerate(features_map), total=len(smiles)):
        temp_features.append(feats)

        # Save temporary features every save_frequency
        if (i > 0 and (i + 1) % save_frequency == 0) or i == len(smiles) - 1:
            save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        # Save all features
        save_features(save_path, features)

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')

# For plot of accuracy metrics
# For regression statistics
def plot_parity(y_true, y_pred, y_pred_unc=None):
    
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    at = AnchoredText(
    f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR-2 = {r2:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel('True')
    plt.ylabel('Chemprop Predicted')
    
    plt.savefig('ParityPlot.png')
    plt.close()
    
    return

# For classification statistics
def plot_auc(y_true, y_pred):
        
    tauc = roc_auc_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_pred)    

    plt.plot(fpr, tpr, marker='.')

    ax = plt.gca()
    ax.set_aspect('equal')
    
    at = AnchoredText(
    f"AUC = {tauc:.2f}", prop=dict(size=10), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('AUCCurve.png')
    plt.close()
    
    return

def plot_precision_recall(y_true, y_pred, y_pred_cat):

    acc = accuracy_score(y_true, y_pred_cat)
    prec = precision_score(y_true, y_pred_cat)
    rec = recall_score(y_true, y_pred_cat)
    f1 = f1_score(y_true, y_pred_cat)
    mcc = matthews_corrcoef(y_true, y_pred_cat)
    precision, recall, _ = precision_recall_curve(y_true, y_pred) 

    rauc = auc(recall, precision)

    plt.plot(recall, precision, marker='.')

    ax = plt.gca()
    ax.set_aspect('equal')
    
    at = AnchoredText(
    f"AUC = {rauc:.2f}\nAccuracy = {acc:.2f}\nPrecision = {prec:.2f}\nRecall = {rec:.2f}\nF1 = {f1:.2f}\nMCC = {mcc:.2f}\n", prop=dict(size=10), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig('PrecisionRecallCurve.png')
    plt.close()
    
    return

def build_model(config):
    """
    Builds model: splits into test train data and performs analysis    
    """
    #split input into test/train data
    data = pd.read_csv(config["input_files"])
    train, test = train_test_split(data, test_size=config['test_size'], random_state=0)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    test.iloc[:,0].to_csv('test_smiles.csv', index=False)

    if config['featurizer'] is not None:
        #Import and save featurised data
        generate_and_save_features(data_path='train.csv', save_path='train.npz', features_generator = config['featurizer'])
        generate_and_save_features(data_path='test.csv', save_path='test.npz', features_generator = config['featurizer'])

    cv_arguments = [
        '--data_path', 'train.csv',
        '--dataset_type', config['type'],
        '--save_dir', 'model',
        '--epochs', config['epochs'],
        '--num_folds', '1'
    ]
    if config['featurizer'] is not None:
        cv_arguments.extend([
            '--features_path', 'train.npz',
            '--no_features_scaling'])

    cv_args = chemprop.args.TrainArgs().parse_args(cv_arguments)
    chemprop.train.cross_validate(args=cv_args, train_func=chemprop.train.run_training)
    
    pred_arguments = [
        '--test_path', 'test_smiles.csv',
        '--preds_path', '/dev/null',
        '--checkpoint_dir', 'model'
    ]
    if config['featurizer'] is not None:
        pred_arguments.extend([
        '--features_path', 'test.npz',
        '--no_features_scaling'])

    pred_args = chemprop.args.PredictArgs().parse_args(pred_arguments)
    preds = chemprop.train.make_predictions(args=pred_args)
    os.remove('test.npz')
    os.remove('train.npz')
    df = pd.read_csv('test.csv')
    df['Predicted_Values'] = [x[0] for x in preds]
    y_true = df.iloc[:,1]
    if config['type']=='regression':
        df.to_csv('Predictions.csv', index=False)
        plot_parity(y_true, df['Predicted_Values'])
        output = {'figures':['ParityPlot.png']}
    else:
        df['Predicted_Categories'] = [round(x[0]) for x in preds]
        df.to_csv('Predictions.csv', index=False)
        plot_auc(y_true, df['Predicted_Values'])
        plot_precision_recall(y_true, df['Predicted_Values'], df['Predicted_Categories'])
        print(confusion_matrix(y_true, df['Predicted_Categories']), '\n')
        output = {'figures':['AUCCurve.png','PrecisionRecallCurve.png']}
    return output


def app_exe(config):
    """
    Runs build model
    """
    output = build_model(config)
    return output

def main():
    config={"input_files": "input.csv", "test_size":0.2, "featurizer":"rdkit_2d_normalized", "type":"classification", "epochs":"10"}

    app_exe(config)


if __name__ ==  '__main__':
    #set correct working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()