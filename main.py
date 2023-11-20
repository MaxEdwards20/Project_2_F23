import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load  # https://scikit-learn.org/stable/model_persistence.html
import matplotlib.pyplot as plt


def plot_predictions(y_real, y_pred, mse, estimators):
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
    # Plot a diagonal line for reference
    # _ = plt.plot(
    #     [-100, 100], [-100, 100], color="red"
    # )
    # plt.show()
    # save the plot
    plt.savefig(f"RF_{estimators}_{mse}.png")


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
    print("Accuracy: ", calc_accuracy(y_test, y_pred))
    mse = calc_mse(y_test, y_pred)
    print("Mean Squared Error: ", mse)
    return mse


def calc_accuracy(y_test, y_pred):
    errors = abs(y_pred - y_test)
    mape = 100 * (errors / (y_test + (y_test == 0)))
    accuracy = abs(100 - np.mean(mape))
    return accuracy


def calc_mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)


def runAnn(
    X_train,
    y_train,
):
    y_train.values.reshape(-1, 1)

    # Bring all features within the same scale
    scaler = MinMaxScaler()
    # this function ensure the data is formatted well for the model to work with
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    # We need to reshape the truth data because the scaler expects a 2D array. TODO: Add link to documentation
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))

    print(X_train_scaled.shape)

    # Build the neural network model using TensorFlow
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                128,
                activation="relu",
                input_shape=(
                    [
                        X_train_scaled.shape[1],
                    ]
                ),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(
                1
            ),  # Output layer, as we're predicting a single value (song popularity)
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    print(" \n--- Training the Model ---\n")
    history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.1)

    # Persist the Model
    model.save("song_predictor.h5")
    return model


def runRandomForest(X_train, y_train, estimators):
    print(f" \n--- Training the Model with {estimators} Trees ---\n")
    filename = "song_predictor_rf_" + str(estimators) + ".joblib"
    # Create a Random Forest Regressor
    startTime = pd.Timestamp.now()
    # Specifc the number of trees to use for estimation, the max depth of each tree and the verbosity for logging
    model = RandomForestRegressor(
        n_estimators=estimators, max_depth=50, verbose=1, random_state=0
    )
    model.fit(X_train, y_train)
    endTime = pd.Timestamp.now()
    print(f"Training took {endTime - startTime} seconds")
    # Persist the Model
    dump(model, filename)
    return model


def main():
    features, target = setup()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.5, random_state=0
    )
    estimators = 2000
    model = runRandomForest(X_train, y_train, estimators)
    # model = load("song_predictor_rf_2000.joblib")
    y_pred = model.predict(X_test)
    mse = evaluateModel(y_pred, y_test)
    # Plot the predictions against the truth
    plot_predictions(y_test, y_pred, mse, estimators)


if __name__ == "__main__":
    main()
