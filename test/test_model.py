# test/test_model.py
from house_price_prediction import load_data, preprocess_data, train_model, evaluate_model

def test_model_training():
    df = load_data("train.csv")
    X, y = preprocess_data(df)
    pipeline, X_test, y_test = train_model(X, y)
    mae, mse, rmse = evaluate_model(pipeline, X_test, y_test)

    assert rmse < 70000, f"❌ RMSE too high: {rmse}"

def test_dummy():
    assert 2 + 2 == 4
