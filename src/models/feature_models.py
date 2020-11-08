import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data.utils import read_dataframe


def lda_classify_workload(config, features):
    # Load the csv
    read_path = config['processed_data']['z_score_psd_power_path']
    df = read_dataframe(read_path)
    scores = []

    for subject in config['subjects']:
        temp = df.loc[df['subject'] == np.asarray(subject, dtype=np.float64)]

        x = temp[features]
        # x = temp[selected_features]
        y = temp['class_label']

        # Perfrom 5 fold cross validation
        clf = LinearDiscriminantAnalysis()
        val_scores = cross_val_score(clf, x, y, cv=KFold(4, shuffle=True))
        scores.append(val_scores.mean() * 100)

    print('\n')
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores)))
    print('\n')
    return np.mean(scores)


def lda_classify_workload_pooled_subjects(config, features):

    # Load the csv
    read_path = config['processed_data']['z_score_psd_power_path']
    df = read_dataframe(read_path)

    x = df[features]
    y = df['class_label']

    # Perfrom 5 fold cross validation
    clf = LinearDiscriminantAnalysis()
    val_scores = cross_val_score(clf, x, y, cv=KFold(4, shuffle=True))
    print("Accuracy: %0.2f (+/- %0.2f)" %
          (val_scores.mean() * 100, val_scores.std() * 100))

    # Original data
    selected_features = [
        'Cz_theta', 'P3_lower_alpha', 'C3_lower_alpha', 'F3_higher_alpha',
        'C3_higher_alpha', 'T4_lower_beta', 'F4_lower_beta', 'P3_lower_beta',
        'T4_higher_beta', 'T3_higher_beta', 'F4_higher_beta', 'C3_gamma',
        'P4_gamma'
    ]
    x = df[selected_features]
    y = df['class_label']

    # Perfrom 5 fold cross validation
    clf = LinearDiscriminantAnalysis()
    val_scores = cross_val_score(clf, x, y, cv=KFold(4, shuffle=True))
    print("Accuracy: %0.2f (+/- %0.2f)" %
          (val_scores.mean() * 100, val_scores.std() * 100))
    return None
