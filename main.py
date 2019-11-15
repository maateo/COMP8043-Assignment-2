############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.metrics import confusion_matrix
from sklearn import svm


def main():
    ####################
    #####  Task 1  #####
    ####################
    data = pd.read_csv("product_images.csv")  # Load the product images and labels

    labels = data["label"]  # Keep only the labels
    feature_vectors = data.drop("label", axis=1)  # Keep the pixel values

    # Print amount of each type of images
    print("There are:"
          "\n\t%d images of sneakers"
          "\n\t%d images of ankle boots"
          % (labels[labels == 0].size,
             labels[labels == 1].size)
          )

    # Get and show first sneaker
    plt.imshow(np.array(feature_vectors.iloc[labels[labels == 0].index[0]]).reshape(28, 28))
    plt.show()

    # Get and show first ankle boot
    plt.imshow(np.array(feature_vectors.iloc[labels[labels == 1].index[0]]).reshape(28, 28))
    plt.show()

    # Parameterised data
    number_of_samples = 400
    feature_vectors_parameterised = feature_vectors.sample(number_of_samples)
    labels_parameterised = labels[feature_vectors_parameterised.index]

    # Print parameterised statistics
    print("Parameterised dataset contains:"
          "\n\t%d images of sneakers"
          "\n\t%d images of ankle boots"
          % (labels_parameterised[labels_parameterised == 0].size,
             labels_parameterised[labels_parameterised == 1].size)
          )

    ####################
    #####  Task 2  #####
    ####################
    number_of_kfolds = 5
    print("#######################")
    print("#     Perceptron      #")
    print("# Number of kfolds: %d #" % number_of_kfolds)
    print("#######################")

    perceptron_training_times = []
    perceptron_prediction_times = []
    perceptron_prediction_accuracies = []
    current_fold = 0
    kf = model_selection.KFold(n_splits=number_of_kfolds, shuffle=True)
    for train_index, test_index in kf.split(feature_vectors_parameterised, labels_parameterised):
        current_fold += 1
        feature_vectors_parameterised_train_fold = feature_vectors_parameterised.iloc[train_index]
        feature_vectors_parameterised_test_fold = feature_vectors_parameterised.iloc[test_index]
        labels_parameterised_train_fold = labels_parameterised.iloc[train_index]
        labels_parameterised_test_fold = labels_parameterised.iloc[test_index]

        perceptron_classifier = linear_model.Perceptron()

        perceptron_fit_start_time = time.time()
        perceptron_classifier.fit(feature_vectors_parameterised_train_fold, labels_parameterised_train_fold)
        perceptron_fit_end_time = time.time()

        perceptron_predict_start_time = time.time()
        perceptron_prediction = perceptron_classifier.predict(feature_vectors_parameterised_test_fold)
        perceptron_predict_end_time = time.time()

        perceptron_accuracy_score = metrics.accuracy_score(labels_parameterised_test_fold, perceptron_prediction)

        perceptron_training_times.append(perceptron_fit_end_time - perceptron_fit_start_time)
        perceptron_prediction_times.append(perceptron_predict_end_time - perceptron_predict_start_time)
        perceptron_prediction_accuracies.append(perceptron_accuracy_score)

        true_negative, false_positive, false_negative, true_positive = confusion_matrix(labels_parameterised_test_fold, perceptron_prediction).ravel()
        print("\t## Fold number: %d ##" % current_fold)
        print("\t\t# Training time", perceptron_fit_end_time - perceptron_fit_start_time)
        print("\t\t# Predicting time", perceptron_predict_end_time - perceptron_predict_start_time)
        print("\t\t# Perceptron accuracy score: ", perceptron_accuracy_score)
        print("\t\t# true negative", true_negative)
        print("\t\t# false positive", false_positive)
        print("\t\t# false negative", false_negative)
        print("\t\t# true positive", true_positive)

    print("### Training Times (in ms) ###")
    print("# Minimum: ", min(perceptron_training_times))
    print("# Maximum: ", max(perceptron_training_times))
    print("# Average: ", np.mean(perceptron_training_times))
    print("### Training Times (in ms) per sample ###")
    print("# Minimum: ", np.divide(min(perceptron_training_times), number_of_samples))
    print("# Maximum: ", np.divide(max(perceptron_training_times), number_of_samples))
    print("# Average: ", np.divide(np.mean(perceptron_training_times), number_of_samples))
    print("### Prediction Times (in ms) ###")
    print("# Minimum: ", min(perceptron_prediction_times))
    print("# Maximum: ", max(perceptron_prediction_times))
    print("# Average: ", np.mean(perceptron_prediction_times))
    print("### Prediction Times (in ms) per sample ###")
    print("# Minimum: ", np.divide(min(perceptron_prediction_times), number_of_samples))
    print("# Maximum: ", np.divide(max(perceptron_prediction_times), number_of_samples))
    print("# Average: ", np.divide(np.mean(perceptron_prediction_times), number_of_samples))
    print("### Accuracies ###")
    print("# Minimum: ", min(perceptron_prediction_accuracies))
    print("# Maximum: ", max(perceptron_prediction_accuracies))
    print("# Average: ", np.mean(perceptron_prediction_accuracies))

    ####################
    #####  Task 3  #####
    ####################
    print()
    print("#########################")
    print("#    Linear Kernel      #")
    print("#         And           #")
    print("# Radial Basis Function #")
    print("#  Number of kfolds: %d  #" % number_of_kfolds)
    print("#########################")

    gamma_rbfc_average_training_times = []
    gamma_rbfc_average_prediction_times = []
    gamma_rbfc_average_accuracies = []

    kf = model_selection.KFold(n_splits=number_of_kfolds, shuffle=True)
    for gamma in range(1, 16):
        print("##### Gamma: %d" % gamma)

        linear_kernel_training_times = []
        linear_kernel_prediction_times = []
        linear_kernel_prediction_accuracies = []
        radial_basis_function_training_times = []
        radial_basis_function_prediction_times = []
        radial_basis_function_prediction_accuracies = []

        current_fold = 0
        for train_index, test_index in kf.split(feature_vectors_parameterised, labels_parameterised):
            current_fold += 1
            feature_vectors_parameterised_train_fold = feature_vectors_parameterised.iloc[train_index]
            feature_vectors_parameterised_test_fold = feature_vectors_parameterised.iloc[test_index]
            labels_parameterised_train_fold = labels_parameterised.iloc[train_index]
            labels_parameterised_test_fold = labels_parameterised.iloc[test_index]

            linear_kernel_classifier = svm.SVC(kernel="linear")
            radial_basis_function_classifier = svm.SVC(kernel="rbf", gamma=float('1e-%d' % gamma))

            linear_kernel_fit_start_time = time.time()
            linear_kernel_classifier.fit(feature_vectors_parameterised_train_fold, labels_parameterised_train_fold)
            linear_kernel_fit_end_time = time.time()
            radial_basis_function_fit_start_time = time.time()
            radial_basis_function_classifier.fit(feature_vectors_parameterised_train_fold, labels_parameterised_train_fold)
            radial_basis_function_fit_end_time = time.time()

            linear_kernel_predict_start_time = time.time()
            linear_kernel_prediction = linear_kernel_classifier.predict(feature_vectors_parameterised_test_fold)
            linear_kernel_predict_end_time = time.time()
            radial_basis_function_predict_start_time = time.time()
            radial_basis_function_prediction = radial_basis_function_classifier.predict(feature_vectors_parameterised_test_fold)
            radial_basis_function_predict_end_time = time.time()

            linear_kernel_accuracy_score = metrics.accuracy_score(labels_parameterised_test_fold, linear_kernel_prediction)
            radial_basis_function_accuracy_score = metrics.accuracy_score(labels_parameterised_test_fold, radial_basis_function_prediction)

            linear_kernel_training_times.append(linear_kernel_fit_end_time - linear_kernel_fit_start_time)
            linear_kernel_prediction_times.append(linear_kernel_predict_end_time - linear_kernel_predict_start_time)
            linear_kernel_prediction_accuracies.append(linear_kernel_accuracy_score)
            radial_basis_function_training_times.append(radial_basis_function_fit_end_time - radial_basis_function_fit_start_time)
            radial_basis_function_prediction_times.append(radial_basis_function_predict_end_time - radial_basis_function_predict_start_time)
            radial_basis_function_prediction_accuracies.append(radial_basis_function_accuracy_score)

            l_true_negative, l_false_positive, l_false_negative, l_true_positive = confusion_matrix(labels_parameterised_test_fold, linear_kernel_prediction).ravel()
            rbf_true_negative, rbf_false_positive, rbf_false_negative, rbf_true_positive = confusion_matrix(labels_parameterised_test_fold, radial_basis_function_prediction).ravel()

            print("\t\t## Fold number: %d ##" % current_fold)
            print("\t\t\t# Linear Kernel")
            print("\t\t\t\t# Training time", linear_kernel_fit_end_time - linear_kernel_fit_start_time)
            print("\t\t\t\t# Predicting time", linear_kernel_predict_end_time - linear_kernel_predict_start_time)
            print("\t\t\t\t# Linear Kernel accuracy score: ", linear_kernel_accuracy_score)
            print("\t\t\t\t# true negative", l_true_negative)
            print("\t\t\t\t# false positive", l_false_positive)
            print("\t\t\t\t# false negative", l_false_negative)
            print("\t\t\t\t# true positive", l_true_positive)
            print("\t\t\t# Radial Basis Function Kernel")
            print("\t\t\t\t# Training time", radial_basis_function_fit_end_time - radial_basis_function_fit_start_time)
            print("\t\t\t\t# Predicting time", radial_basis_function_predict_end_time - radial_basis_function_predict_start_time)
            print("\t\t\t\t# RBF Kernel accuracy score: ", radial_basis_function_accuracy_score)
            print("\t\t\t\t# true negative", rbf_true_negative)
            print("\t\t\t\t# false positive", rbf_false_positive)
            print("\t\t\t\t# false negative", rbf_false_negative)
            print("\t\t\t\t# true positive", rbf_true_positive)

        gamma_rbfc_average_training_times.append(np.mean(radial_basis_function_training_times))
        gamma_rbfc_average_prediction_times.append(np.mean(radial_basis_function_prediction_times))
        gamma_rbfc_average_accuracies.append(np.mean(radial_basis_function_prediction_accuracies))

        print("\t##### Linear Kernel")
        print("\t\t### Training Times (in ms) ###")
        print("\t\t\t# Minimum: ", min(linear_kernel_training_times))
        print("\t\t\t# Maximum: ", max(linear_kernel_training_times))
        print("\t\t\t# Average: ", np.mean(linear_kernel_training_times))
        print("\t\t### Training Times (in ms) per sample ###")
        print("\t\t\t# Minimum: ", np.divide(min(linear_kernel_training_times), number_of_samples))
        print("\t\t\t# Maximum: ", np.divide(max(linear_kernel_training_times), number_of_samples))
        print("\t\t\t# Average: ", np.divide(np.mean(linear_kernel_training_times), number_of_samples))
        print("\t\t### Prediction Times (in ms) ###")
        print("\t\t\t# Minimum: ", min(linear_kernel_prediction_times))
        print("\t\t\t# Maximum: ", max(linear_kernel_prediction_times))
        print("\t\t\t# Average: ", np.mean(linear_kernel_prediction_times))
        print("\t\t### Prediction Times (in ms) per sample ###")
        print("\t\t\t# Minimum: ", np.divide(min(linear_kernel_prediction_times), number_of_samples))
        print("\t\t\t# Maximum: ", np.divide(max(linear_kernel_prediction_times), number_of_samples))
        print("\t\t\t# Average: ", np.divide(np.mean(linear_kernel_prediction_times), number_of_samples))
        print("\t\t### Accuracies ###")
        print("\t\t\t# Minimum: ", min(linear_kernel_prediction_accuracies))
        print("\t\t\t# Maximum: ", max(linear_kernel_prediction_accuracies))
        print("\t\t\t# Average: ", np.mean(linear_kernel_prediction_accuracies))
        print("\t##### Radial Basis Function Kernel")
        print("\t\t### Training Times (in ms) ###")
        print("\t\t\t# Minimum: ", min(radial_basis_function_training_times))
        print("\t\t\t# Maximum: ", max(radial_basis_function_training_times))
        print("\t\t\t# Average: ", np.mean(radial_basis_function_training_times))
        print("\t\t### Training Times (in ms) per sample ###")
        print("\t\t\t# Minimum: ", np.divide(min(radial_basis_function_training_times), number_of_samples))
        print("\t\t\t# Maximum: ", np.divide(max(radial_basis_function_training_times), number_of_samples))
        print("\t\t\t# Average: ", np.divide(np.mean(radial_basis_function_training_times), number_of_samples))
        print("\t\t### Prediction Times (in ms) ###")
        print("\t\t\t# Minimum: ", min(radial_basis_function_prediction_times))
        print("\t\t\t# Maximum: ", max(radial_basis_function_prediction_times))
        print("\t\t\t# Average: ", np.mean(radial_basis_function_prediction_times))
        print("\t\t### Prediction Times (in ms) per sample ###")
        print("\t\t\t# Minimum: ", np.divide(min(radial_basis_function_prediction_times), number_of_samples))
        print("\t\t\t# Maximum: ", np.divide(max(radial_basis_function_prediction_times), number_of_samples))
        print("\t\t\t# Average: ", np.divide(np.mean(radial_basis_function_prediction_times), number_of_samples))
        print("\t\t### Accuracies ###")
        print("\t\t\t# Minimum: ", min(radial_basis_function_prediction_accuracies))
        print("\t\t\t# Maximum: ", max(radial_basis_function_prediction_accuracies))
        print("\t\t\t# Average: ", np.mean(radial_basis_function_prediction_accuracies))

    np.set_printoptions(precision=5, linewidth=200)
    print("##### Summary of gammas for radial basis function #####")
    print("\t# Average training times for each gamma:\t", np.array2string(np.array(gamma_rbfc_average_training_times), separator=', '))
    print("\t# Average predicting times for each gamma:\t", np.array2string(np.array(gamma_rbfc_average_prediction_times), separator=', '))
    print("\t# Average accuracies for each gamma:\t\t", np.array2string(np.array(gamma_rbfc_average_accuracies), separator=', '))
    print("\t# Best gamma in terms of training time:", gamma_rbfc_average_training_times.index(min(gamma_rbfc_average_training_times)) + 1)  # Min because faster is better
    print("\t# Best gamma in terms of predicting time:", gamma_rbfc_average_prediction_times.index(min(gamma_rbfc_average_prediction_times)) + 1)  # Min because faster is better
    print("\t# Best gamma in terms of accuracy:", gamma_rbfc_average_accuracies.index(max(gamma_rbfc_average_accuracies)) + 1)  # Max because we want highest accuracy possible


main()
