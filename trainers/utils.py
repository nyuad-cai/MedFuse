import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_new(df):
    auroc = roc_auc_score(df['y_truth'], df['y_pred'])
    auprc = average_precision_score(df['y_truth'], df['y_pred'])
    return auprc, auroc



def bootstraping_eval(df, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    auroc_list = []
    auprc_list = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
    return auprc_list, auroc_list

def computing_confidence_intervals(list_,true_value):
    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    # print(f"CI 95% {round(true_value, 3)} ( {round(lower, 3)} , {round(upper, 3)} )")
    return (upper,lower)


def get_model_performance(df):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=1000)
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc)
