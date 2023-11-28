import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load  # https://scikit-learn.org/stable/model_persistence.html
from aux_funs import setup, evaluateModel, plot_predictions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def runRandomForest(X_train, y_train, estimators):
    print(f" \n--- Training the Model with {estimators} Trees ---\n")
    filename = "song_predictor_rf_" + str(estimators) + ".joblib"
    startTime = pd.Timestamp.now()
    # Create a Random Forest Regressor
    # Specifc the number of trees to use for estimation, the max depth of each tree and the verbosity for logging
    model = RandomForestRegressor(n_estimators=estimators, verbose=1, random_state=0)
    model.fit(X_train, y_train)
    endTime = pd.Timestamp.now()
    print(f"Training took {endTime - startTime} seconds")
    # Persist the Model
    dump(model, filename)
    return model


def runPolyRegression(features, target):
    # First we need to reshape the predictions from the random forest
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        features, target, test_size=0.2, random_state=0
    )
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_poly)
    # Now we train a linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train_poly)
    X_test_poly = poly.fit_transform(X_test_poly)
    # Now we train a polynomial regression model
    y_pred_poly = model.predict(X_test_poly)
    return y_pred_poly, y_test_poly


def main():
    loadModel = False
    features, target = setup()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0
    )
    estimators = 1000

    if loadModel:
        # put name of model you want to load here
        model = load("./persistedModels/song_predictor_rf_2000.joblib")
    else:
        # We are training the model
        model = runRandomForest(X_train, y_train, estimators)
    y_pred = model.predict(X_test)

    print("\nEvaluating random forest model")
    mse = evaluateModel(y_pred, y_test)
    # Plot the predictions against the truth
    plot_predictions(y_test, y_pred, mse, f"RF_{estimators}.png")

    # Now we use a polynomial regression for the output of the random forest
    # We use the predictions from the random forest as the input to the polynomial regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    # Now we output all of the data from the random forest for the polynomial regression to have something to work on
    predictions = model.predict(features)
    predictions = np.array(predictions).reshape(-1, 1)  # reshape to 2D array
    y_pred_poly, y_test_poly = runPolyRegression(predictions, target)
    print("\nEvaluating polynomial regression model")
    mse = evaluateModel(y_pred_poly, y_test_poly)
    # Plot the predictions against the truth
    plot_predictions(y_test_poly, y_pred_poly, mse, f"RF_LR_{estimators}.png")


if __name__ == "__main__":
    main()
