############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv("product_images.csv")


    figures = []
    for index, row in data.iterrows():
        figures.append(plt.imshow(row[1:].to_numpy().reshape(28, 28)))

        if index == 10:
            break

    print(type(figures[0]))
    # figures[0].show()
    # print("FIGURE:", figures[0])
    # Try doing it liek the other oen.... except that we convert the data into a numpy array with rows and then that into the pictures stuff
    plt.show()
    raise SystemExit


    labels = data["label"]
    print(labels)

    print(plt.hist)
    plt.figure(20).show()



    print("There are:"
          "\n\t%d images of sneakers"
          "\n\t%d images of ankle boots"
          % (data["label"][data["label"] == 0].size,
             data["label"][data["label"] == 1].size)
          )

    # plt.imshow(data[1, data[1:]].reshape(28, 28))
    #
    # print(data)


main()
