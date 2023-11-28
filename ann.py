import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from aux_funs import setup, evaluateModel, plot_predictions
import numpy as np


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
    # We need to reshape the truth data because the scaler expects a 2D array.
    #  TODO: Add link to documentation
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
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
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
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


def main():
    loadModel = False
    features, target = setup()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0
    )
    if loadModel:
        # load the song_predictor.h5 model
        model = tf.keras.models.load_model("song_predictor.h5")
    else:
        # We are training the model
        model = runAnn(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = evaluateModel(y_pred, y_test)
    # Calculate the mean squared error
    # Plot the predictions against the truth
    plot_predictions(y_test, y_pred, mse, "ann.png")


if __name__ == "__main__":
    main()
