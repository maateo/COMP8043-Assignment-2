############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv("product_images.csv")  # Load the product images and labels

    labels = data["label"]  # Keep only the labels
    feature_vectors = data.drop("label", axis=1)  # Keep the pixel values
    print(labels.head())
    print(type(labels))
    print(feature_vectors.head())
    print(type(feature_vectors))

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
    feature_vectors_parameterised = feature_vectors.sample(400)
    labels_parameterised = labels[feature_vectors_parameterised.index]
    print("Parameterised dataset contains:"
          "\n\t%d images of sneakers"
          "\n\t%d images of ankle boots"
          % (labels_parameterised[labels_parameterised == 0].size,
             labels_parameterised[labels_parameterised == 1].size)
          )


main()
