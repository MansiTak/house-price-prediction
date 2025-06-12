import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# test/test_model.py

from house_price_prediction import load_data, preprocess_data, train_model, evaluate_model, example_addition


def test_example_addition():
    assert example_addition(2, 3) == 5
    assert example_addition(-1, 1) == 0


def test_pipeline_evaluation():
    df = load_data("train.csv")
    X, y = preprocess_data(df)
    pipeline, X_test, y_test = train_model(X, y)
    mae, mse, rmse, y_pred = evaluate_model(pipeline, X_test, y_test)

    assert rmse < 80000, f"❌ RMSE too high: {rmse}"
    assert len(y_pred) == len(y_test), "❌ Prediction count mismatch"

def test_dummy():
    assert 2 + 2 == 4
