import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup():
    # Load the data
    data = pd.read_csv("song_data.csv")
    # Select relevant columns for input (features) and output (target)
    features = data[["song_duration_ms", "danceability", "loudness", "tempo"]]
    target = data["song_popularity"]
    return features, target


def evaluateModel(y_pred, y_test):
    # Evaluate the model on the test set
    print(" \n--- Evaluating the Model ---\n")
    closeness = 5
    print(
        f"Accuracy within {closeness} popularity levels: {round(calc_accuracy(y_test, y_pred, closeness) * 100, 2)}%",
    )
    closeness = 10
    print(
        f"Accuracy within {closeness} popularity levels: {round(calc_accuracy(y_test, y_pred, closeness) * 100, 2)}%",
    )
    closeness = 25
    print(
        f"Accuracy within {closeness} popularity levels: {round(calc_accuracy(y_test, y_pred, closeness) * 100, 2)}%",
    )
    mse = calc_mse(y_test, y_pred)
    print("Mean Squared Error: ", mse)
    return mse


def calc_mse(y_test, y_pred):
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    return np.mean((y_test - y_pred) ** 2)


def calc_accuracy(y_test, y_pred: np.ndarray, closeness: int):
    """
    Calculates the accuracy of the model by checking how many predictions are within X of the true value
    :param y_test: The true values
    :param y_pred: The predicted values
    :param closeness: The closeness to check for
    :return: The accuracy of the model in the range [0, 1]
    """
    # Find the accuracy of the model
    within_closeness = 0
    y_test = y_test.values
    for i in range(len(y_pred)):
        # Cleaup the predictions between 0 and 100
        if y_pred[i] < 0:
            y_pred[i] = 0
        elif y_pred[i] > 100:
            y_pred[i] = 100
        # Now check if it is within 5 of the true value
        if abs(y_pred[i] - y_test[i]) <= closeness:
            within_closeness += 1
    return within_closeness / len(y_pred)


def plot_predictions(y_real, y_pred, mse, filename):
    mse = round(mse, 2)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.5, label="Predictions", color="blue")
    plt.scatter(y_real, y_real, alpha=0.5, label="True Values", color="orange")
    plt.xlabel("True Values [Song Popularity]")
    plt.ylabel("Predictions [Song Popularity]")
    plt.title(f"Song Popularity Predictions MSE: {mse}")
    plt.axis("equal")
    plt.axis("square")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend()
    # save the plot
    plt.savefig(filename)
