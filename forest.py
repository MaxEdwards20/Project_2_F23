import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load  # https://scikit-learn.org/stable/model_persistence.html
from aux_funs import setup, evaluateModel, plot_predictions


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
    loadModel = False
    features, target = setup()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0
    )
    estimators = 2000

    if loadModel:
        # put name of model you want to load here
        model = load("song_predictor_rf_500.joblib")
    else:
        # We are training the model
        model = runRandomForest(X_train, y_train, estimators)
    y_pred = model.predict(X_test)
    mse = evaluateModel(y_pred, y_test)
    # Plot the predictions against the truth
    plot_predictions(y_test, y_pred, mse, f"RF_{estimators}.png")


if __name__ == "__main__":
    main()
