from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_conf_matrix_by_filename(filename):
    dataframe = pd.read_csv(filename)
    true_class = dataframe['actual']
    predicted_class = dataframe['predicted']
    print(true_class)
    print(predicted_class)
    labels = ["transfer intragene intraspecies", "transfer intragene interspecies", "transfer intergene intraspecies", "transfer intergene interspecies"]
    multilabel_confusion_matrix = confusion_matrix(true_class, predicted_class,
                                                   labels=labels)
    print(multilabel_confusion_matrix)
    class_names = ['wg ws', 'wg is', 'ig ws', 'ig is']
    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names = class_names
                                    )
    #plt.xticks(np.arange(0,4), labels=['wg ws', 'wg is', 'ig ws', 'ig is'])

    #ax.set(yticks=[ -0.5, 0, 0.5, 1, 1.5, 2, 2.5],

    #       yticklabels=['', 'wg ws', '', 'wg is', '', 'ig ws', ])

    plt.show()

def plot_binary_conf_matrix_by_filename(filename):
    dataframe = pd.read_csv(filename)
    true_class = dataframe['actual']
    predicted_class = dataframe['predicted']
    print(true_class)
    print(predicted_class)
    labels = ["intraspecies", "interspecies"]
    multilabel_confusion_matrix = confusion_matrix(true_class, predicted_class,
                                                   labels=labels)
    print(multilabel_confusion_matrix)
    class_names = ["intraspecies", "interspecies"]
    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=class_names
                                    )
    # plt.xticks(np.arange(0,4), labels=['wg ws', 'wg is', 'ig ws', 'ig is'])

    # ax.set(yticks=[ -0.5, 0, 0.5, 1, 1.5, 2, 2.5],

    #       yticklabels=['', 'wg ws', '', 'wg is', '', 'ig ws', ])

    plt.show()
#plot_conf_matrix_by_filename('Approximation_transfer.txt')
#plot_conf_matrix_by_filename('ExtendedDGSDTL_transfer.txt')
plot_binary_conf_matrix_by_filename('Approximation_species_transfer.txt')
plot_binary_conf_matrix_by_filename('ExtendedDGSDTL_species_transfer.txt')