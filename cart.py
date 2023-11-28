import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from aux_funs import setup, evaluateModel, plot_predictions
from joblib import dump, load  # https://scikit-learn.org/stable/model_persistence.html


def runCart(X_train, y_train):
    # Creating and training the CART model
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    cart_model = DecisionTreeRegressor(random_state=0)
    cart_model.fit(X_train, y_train)
    # Persist the Model
    dump(cart_model, "cart.joblib")
    return cart_model


def main():
    loadModel = True
    features, target = setup()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0
    )

    if loadModel:
        # put name of model you want to load here
        model = load("./persistedModels/cart.joblib")
    else:
        # We are training the model
        model = runCart(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = evaluateModel(y_pred, y_test)
    # Plot the predictions against the truth
    plot_predictions(y_test, y_pred, mse, f"cart.png")


if __name__ == "__main__":
    main()
