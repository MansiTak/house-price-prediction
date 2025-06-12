import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np  # Import numpy

# Load the trained model pipeline and input template
pipeline = joblib.load("model_pipeline.pkl")
template = pd.read_csv("input_template.csv")

# Set page config for better layout
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Title
st.title("🏠 House Price Prediction App")

# Create two tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Evaluation Results", "📂 View Dataset"])

# -------------------------
# 🔮 Tab 1: Predict Price
# -------------------------
with tab1:
    st.write("### Enter House Features Below:")

    # ===== Common numerical inputs =====
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, int(template.loc[0, 'OverallQual']))
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 100, 10000, int(template.loc[0, 'GrLivArea']))
    garage_cars = st.slider("Garage Cars Capacity", 0, 5, int(template.loc[0, 'GarageCars']))
    garage_area = st.number_input("Garage Area (sq ft)", 0, 2000, int(template.loc[0, 'GarageArea']))
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, int(template.loc[0, 'TotalBsmtSF']))
    year_built = st.slider("Year Built", 1870, 2025, int(template.loc[0, 'YearBuilt']))

    # ===== Other useful numeric fields (optional) =====
    first_flr_sf = st.number_input("1st Floor SF", 0, 4000, int(template.loc[0, '1stFlrSF']))
    second_flr_sf = st.number_input("2nd Floor SF", 0, 4000, int(template.loc[0, '2ndFlrSF']))
    full_bath = st.slider("Full Bathrooms", 0, 5, int(template.loc[0, 'FullBath']))
    half_bath = st.slider("Half Bathrooms", 0, 3, int(template.loc[0, 'HalfBath']))
    totrms_abvgrd = st.slider("Total Rooms Above Ground", 1, 15, int(template.loc[0, 'TotRmsAbvGrd']))
    fireplaces = st.slider("Fireplaces", 0, 4, int(template.loc[0, 'Fireplaces']))
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, int(template.loc[0, 'LotArea']))

    # ===== Update only selected fields in the full template =====
    input_data = template.copy()
    input_data.at[0, 'OverallQual'] = overall_qual
    input_data.at[0, 'GrLivArea'] = gr_liv_area
    input_data.at[0, 'GarageCars'] = garage_cars
    input_data.at[0, 'GarageArea'] = garage_area
    input_data.at[0, 'TotalBsmtSF'] = total_bsmt_sf
    input_data.at[0, 'YearBuilt'] = year_built
    input_data.at[0, '1stFlrSF'] = first_flr_sf
    input_data.at[0, '2ndFlrSF'] = second_flr_sf
    input_data.at[0, 'FullBath'] = full_bath
    input_data.at[0, 'HalfBath'] = half_bath
    input_data.at[0, 'TotRmsAbvGrd'] = totrms_abvgrd
    input_data.at[0, 'Fireplaces'] = fireplaces
    input_data.at[0, 'LotArea'] = lot_area

    # Predict button
    if st.button("Predict Price"):
        prediction = pipeline.predict(input_data)[0]
        st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")

# -------------------------
# 📊 Tab 2: Evaluation Results
# -------------------------
with tab2:
    st.write("### Model Evaluation on Test Set")

    # Load y_test and y_pred
    y_test = pd.read_csv("y_test.csv")
    y_pred = pd.read_csv("y_pred.csv")

    # Ensure both y_test and y_pred are Series and reset their indices
    y_test = y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    # Convert y_test and y_pred to numpy arrays
    y_test = np.array(y_test).flatten()  # Convert to 1D array
    y_pred = np.array(y_pred).flatten()  # Convert to 1D array

    # Debugging: Print the length of y_test and y_pred
    st.write(f"Length of y_test: {len(y_test)}")
    st.write(f"Length of y_pred: {len(y_pred)}")

    # Check that the lengths match
    if len(y_test) != len(y_pred):
        st.error("Mismatch between the lengths of actual and predicted values!")
    else:
        # Plot 1: Actual vs Predicted
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
        ax1.set_xlabel("Actual Prices")
        ax1.set_ylabel("Predicted Prices")
        ax1.set_title("Actual vs Predicted House Prices")
        st.pyplot(fig1)

        # Plot 2: Residuals vs Predicted
        residuals = y_test - y_pred
        
        # Debugging: Print the length of residuals
        st.write(f"Length of residuals: {len(residuals)}")

        # Check if residuals and y_pred have the same length
        if len(residuals) != len(y_pred):
            st.error("Mismatch between the lengths of residuals and predicted values!")
        else:
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred, residuals, alpha=0.5, color='purple')
            ax2.axhline(0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel("Predicted Prices")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residuals vs Predicted Prices")
            st.pyplot(fig2)


# -------------------------
# 📂 Tab 3: View Dataset
# -------------------------
with tab3:
    st.write(" Dataset Preview","from Kaggle")
    data = pd.read_csv("train.csv")  # Replace with the actual filename
    st.dataframe(data)
