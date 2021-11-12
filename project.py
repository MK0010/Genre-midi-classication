import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def find_nan_columns(frame):
    a = frame.isna().sum()
    a = np.array(a)
    return np.argwhere(a > 0).tolist()

def get_data_frame(path, nrows=None):
    frame = pd.read_csv(path, nrows=nrows)
    cols = frame.columns
    
    #print(cols[find_nan_columns(frame)[0]])


    frame.drop(cols[find_nan_columns(frame)[0]], axis=1, inplace=True)

    frame.drop('path', axis=1, inplace=True)
    genre_list = frame['genre'].unique()
    #print(genre_list)
    return frame, genre_list

def contingency(df, col_1, col_2):
    dict_1 = {x : i for i, x in enumerate(set(df.iloc[:, col_1]))}
    dict_2 = {x : i for i, x in enumerate(set(df.iloc[:, col_2]))}

    res = np.zeros((len(dict_1), len(dict_2)), dtype=int)
    #print(col_1, col_2)
    for i in range(df.shape[0]):
        res[dict_1[df.iloc[i, col_1]], dict_2[df.iloc[i, col_2]]] += 1

    return res

def run_chi2(frame, override=False):
    if not override:
        chi = open("chi2.txt")
        chi2_list = []
        for num in chi.read().split('\n'):
            chi2_list.append(float(num))
        return chi2_list

    chi2_list = [0] * (frame.shape[1] - 2)
    for f in range(2, frame.shape[1]):
        chi2, p, dof, ex = scipy.stats.chi2_contingency(contingency(frame, 1, f))
        chi2_list[f - 2] = chi2
    with open('chi2.txt', 'w') as f:
        f.write('\n'.join(list(map(str, chi2_list))))
    return chi2_list

def reduce_prop_genre_size(x_arr, y_arr, genre_list, prop=1):
    genres_idx = {genre : [] for genre in genre_list}

    for i in range(len(y_arr)):
        genres_idx[y_arr[i]].append(i)
    
    min_size_genre = 999999
    for genre in genre_list:
        if len(genres_idx[genre]) < min_size_genre:
            min_size_genre = len(genres_idx[genre])
    
    idx_to_keep = []
    for genre in genre_list:
        random.shuffle(genres_idx[genre])
        idx_to_keep += genres_idx[genre][:min(len(genres_idx[genre]), int(prop*min_size_genre))]

    new_x_arr = x_arr[idx_to_keep]
    new_y_arr = y_arr[idx_to_keep]
    return new_x_arr, new_y_arr

def train_forest_and_select_features(x_train, y_train, x_test, y_test, features_list="", num_features_to_keep=10, to_plot=False):
    # Train random forest
    sel = RandomForestClassifier(n_estimators = 500)
    sel.fit(x_train, y_train)

    y_pred = sel.predict(x_test)
    #print("Confusion matrix of the random forest :")
    #print(confusion_matrix(y_test, y_pred))
    #print("Accuracy of the random forest")
    #print(accuracy_score(y_test, y_pred))

    best_features_idx = sel.feature_importances_.argsort()[::-1][0:num_features_to_keep]

    #sfm = SelectFromModel(sel, max_features=num_features_to_keep, threshold=-np.inf)
    #sfm.fit(x_train, y_train)
    #best_features_idx = sfm.get_support(indices=True)

    if to_plot:
        plt.figure()
        plt.barh(features_list[best_features_idx], sel.feature_importances_[best_features_idx])
        plt.title("Features importance in random forest")
    
    return sel, best_features_idx

def show_PCA(x_arr, y_arr, genre_list, show_2D=True, show_3D=True, title=""):
    pca = PCA(n_components=3, whiten=True)
    pca.fit(x_arr)
    x_pca = pca.transform(x_arr)
    pca_0 = x_pca[:, 0]
    pca_1 = x_pca[:, 1]

    colors_palette = ['red', 'green', 'blue', 'yellow', 'cyan', 'black', 'purple', 'orange', 'pink'][:len(genre_list)]

    if(show_2D):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(pca_0, pca_1, hue=y_arr, palette=colors_palette)
        plt.title(title, pad=15)
        plt.xlabel("1st component")
        plt.ylabel("2nd component")

    if(show_3D):
        Xax = x_pca[:,0]
        Yax = x_pca[:,1]
        Zax = x_pca[:,2]
        colors = colors_palette
        cdict = {genre_list[i] : colors[i] for i in range(len(genre_list))}

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        plt.title(title)

        fig.patch.set_facecolor('white')
        for label in genre_list:
            ix=np.where(y_arr==label)
            ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[label], s=40,
                    label=label)

        ax.set_xlabel("First Principal Component", fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component", fontsize=14)
        ax.legend()

    return

@ignore_warnings(category=ConvergenceWarning)
def run_ga(x_arr, y_arr, max_features):
    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(estimator,
                                    cv=5,
                                    verbose=0,
                                    scoring="accuracy",
                                    max_features=max_features,
                                    n_population=50,
                                    crossover_proba=0.5,
                                    mutation_proba=0.2,
                                    n_generations=20,
                                    crossover_independent_proba=0.5,
                                    mutation_independent_proba=0.05,
                                    tournament_size=3,
                                    n_gen_no_change=5,
                                    caching=True,
                                    n_jobs=4)

    selector = selector.fit(x_arr, y_arr)

    best_features_idx = []

    for i in range(len(selector.support_)):
        if selector.support_[i]:
            best_features_idx.append(i)

    return best_features_idx

@ignore_warnings(category=ConvergenceWarning)
def train_model(x_train, x_test, y_train, y_test, to_plot=False):
    # Neural Network and SVM Configurations
    #clf_mlp = MLPClassifier(solver='adam', learning_rate='adaptive', learning_rate_init=5e-3, alpha=1e-5, hidden_layer_sizes=(10, 5, 5, 5), random_state=1, max_iter=200, warm_start=False)
    #for i in range(10):
    #clf_mlp = MLPClassifier(solver='adam', learning_rate='adaptive', learning_rate_init=5e-5, alpha=1e-5, hidden_layer_sizes=(100), max_iter=100, random_state=1)

    clf_mlp = MLPClassifier(alpha=1e-4, hidden_layer_sizes=(50,), random_state=1, max_iter=5, warm_start=True)
    for i in range(50):
        clf_mlp.fit(x_train, y_train)

    y_pred_mlp = clf_mlp.predict(x_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    if(to_plot):
        print(confusion_matrix(y_test, y_pred_mlp).T)
        print("Accuracy MLP", accuracy_mlp)
        
    clf_svm = SVC(C=1e-1, gamma=1e-1)
    clf_svm.fit(x_train, y_train)
    y_pred_svm = clf_svm.predict(x_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    if(to_plot):
        print("Accuracy SVM", accuracy_svm)
    return round(accuracy_mlp, 5), round(accuracy_svm, 5)

def remove_genre(frame, genre_idx, genre_list):
    for idx in genre_idx:
        frame = frame.drop(frame[frame["genre"] == genre_list[idx]].index)
    genre_list = np.delete(genre_list, genre_idx)
    return frame, genre_list

def run_main(removed_genres,accuracies, num_features_to_keep=15, prop_genre=1.5, to_plot=False):
    frame, genre_list = get_data_frame("save_extracted.csv", nrows=None)
    for i in range(len(genre_list)):
        if(to_plot):
            print(i, genre_list[i], np.sum(frame['genre'] == genre_list[i]))
    
    accuracies['removed'].append(removed_genres)

    # Remove some genres
    frame, genre_list = remove_genre(frame, removed_genres, genre_list)

    # Take features as numpy
    x = frame.iloc[: , 1:]
    feature_list = x.columns
    x_arr = x.to_numpy()

    # Take labels as numpy
    y = frame.iloc[: , :1]
    cols_y = y.columns
    y_arr = y.to_numpy().T[0]

    #x_arr = x_arr[:, random.sample(range(x_arr.shape[1]), 500)]
    human_keep = ['Number of Pitches', 'Number of Common Pitches', 'Range', 'Pitch Variability', 'Repeated Notes', 'Chord Duration', 'Total Number of Notes', 'Mean Rhythmic Value', 'Most Common Rhythmic Value', 'Variability of Time Between Attacks', 'Complex Chords']
    bext_features_idx_human = []
    for to_keep in human_keep:
        bext_features_idx_human.append(list(feature_list).index(to_keep))

    # Normalize
    x_arr = StandardScaler().fit_transform(x_arr)

    # Avoid aving one majoritary genre
    x_arr, y_arr = reduce_prop_genre_size(x_arr, y_arr, genre_list, prop=1.5)
    if(len(y_arr)/len(genre_list) < 100):
        return None
    if(True):
        print(len(y_arr), "data kept over", len(genre_list), "genres", x_arr.shape)

    # Split train test
    x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.3, random_state=42)

    # Run chi2
    max_features = num_features_to_keep
    if(to_plot):
        print("\n---------\nMost influent features with chi2 :")
    chichi = np.abs(np.array(run_chi2(frame)))
    print(chichi)
    #best_features_idx_chi = chichi.argsort()[::-1][0:max_features]
    best_features_idx_chi = list(range(max_features))
    if(to_plot):
        print(feature_list[best_features_idx_chi].to_list())
        plt.figure()
        plt.barh(feature_list[best_features_idx_chi], chichi[best_features_idx_chi])
        plt.title("Features importance in chi2")

    # Train random forest and get best features
    if(to_plot):
        print("\n---------\nMost influent features with trees :")
    _, best_features_idx_forest = train_forest_and_select_features(x_train, y_train, x_test, y_test, feature_list, num_features_to_keep=num_features_to_keep, to_plot=to_plot)
    if(to_plot):
        print(feature_list[best_features_idx_forest].to_list())

    # Run GA
    if(to_plot):
        print("\n---------\nMost influent features with genetic :")
    best_features_idx = run_ga(x_arr, y_arr, max_features=num_features_to_keep)
    if(to_plot):
        print(feature_list[best_features_idx].to_list())


    print(x_train[: , best_features_idx_chi].shape)
    print(x_train[: , best_features_idx_forest].shape)
    print(x_train[: , best_features_idx].shape)
    # Run MLP
    if(to_plot):
        print("\n---------\nResult with chi2 feature selection")

    acc_mlp, acc_svm = train_model(x_train[: , best_features_idx_chi], x_test[: , best_features_idx_chi], y_train, y_test, to_plot=to_plot)
    accuracies['chi2'].append(acc_mlp)
    if(to_plot):
        print("\n---------\nResult with tree feature selection")
    acc_mlp, acc_svm = train_model(x_train[: , best_features_idx_forest], x_test[: , best_features_idx_forest], y_train, y_test, to_plot=to_plot)
    accuracies['without'].append(acc_mlp)
    if(to_plot):
        print("\n---------\nResult with genetic feature selection")
    acc_mlp, acc_svm = train_model(x_train[: , best_features_idx], x_test[: , best_features_idx], y_train, y_test, to_plot=to_plot)
    accuracies['genetic'].append(acc_mlp)
    if(to_plot):
        print("\n---------\nResult without selection")
    acc_mlp, acc_svm = train_model(x_train, x_test, y_train, y_test, to_plot=to_plot)
    accuracies['tree'].append(acc_mlp)

    if(to_plot):
        print("\n---------\nResult with human selection")
    acc_mlp, acc_svm = train_model(x_train[:, bext_features_idx_human], x_test[:, bext_features_idx_human], y_train, y_test, to_plot=to_plot)
    accuracies['tree'].append(acc_mlp)

    for idx in best_features_idx_forest:
        if idx in best_features_idx:
            print(idx)

    if(to_plot):
        # Show PCA before selection
        show_PCA(x_arr, y_arr, genre_list, show_2D=True, show_3D=False, title="PCA before feature selection")

        # Show PCA after selection
        show_PCA(x_arr[:,best_features_idx], y_arr, genre_list, show_2D=True, show_3D=False, title="PCA after genetic feature selection")

        # Show PCA after tree selection
        show_PCA(x_arr[:,best_features_idx_forest], y_arr, genre_list, show_2D=True, show_3D=False, title="PCA after tree based feature selection")

        # Show PCA after chi selection
        show_PCA(x_arr[:,best_features_idx_chi], y_arr, genre_list, show_2D=True, show_3D=False, title="PCA after chi based feature selection")

        # Show PCA after human selection
        show_PCA(x_arr[:,bext_features_idx_human], y_arr, genre_list, show_2D=True, show_3D=False, title="PCA after human feature selection")

        plt.show()

    return accuracies

import json

def main():
    accuracies = {'chi2' : [], 'without' : [], 'genetic' : [], 'tree' : [], 'removed' : []}


    accuracies = {'chi2' : [], 'without' : [], 'genetic' : [], 'tree' : [], 'removed' : []}
    accuracies = run_main([0, 1, 2, 4, 5, 8, 9, 11, 12], accuracies, num_features_to_keep=10, prop_genre=1.8, to_plot=True)

    # Trying all genres

    accuracies = {'chi2' : [], 'without' : [], 'genetic' : [], 'tree' : [], 'removed' : []}

    f_out = open('my_output.txt', 'a')
    f_out.write(json.dumps(accuracies))
    f_out.close()

    for a in range(1, 13-3 - 10):
        for b in range(a+1, 13-2):
            for c in range(b+1, 13-1):
                for d in range(c+1, 13):
                    removed_genres = list(range(13))
                    genre_to_keep = [a, b, c, d]

                    for genre in genre_to_keep:
                        if genre in removed_genres:
                            removed_genres.remove(genre)
                    accuracies = {'chi2' : [], 'without' : [], 'genetic' : [], 'tree' : [], 'removed' : []}
                    accuracies = run_main(removed_genres, accuracies, num_features_to_keep=20, to_plot=False)
                    if accuracies != None:
                        f_out = open('my_output.txt', 'a')
                        f_out.write(json.dumps(accuracies))
                        f_out.write('\n')
                        f_out.close()


if __name__ == "__main__":
    main()
