import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("winequality.csv")

# Split features and target variable
X = data.drop('quality', axis=1)
y = data['quality']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict wine quality
def predict_wine_quality():
    try:
        # Get input values from entry fields
        fixed_acidity = float(fixed_acidity_entry.get())
        volatile_acidity = float(volatile_acidity_entry.get())
        citric_acid = float(citric_acid_entry.get())
        residual_sugar = float(residual_sugar_entry.get())
        chlorides = float(chlorides_entry.get())
        free_sulfur_dioxide = float(free_sulfur_dioxide_entry.get())
        total_sulfur_dioxide = float(total_sulfur_dioxide_entry.get())
        density = float(density_entry.get())
        pH = float(pH_entry.get())
        sulphates = float(sulphates_entry.get())
        alcohol = float(alcohol_entry.get())

        # Create input array
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        # Standardize the input data
        input_data = scaler.transform(input_data)

        # Shuffle the order of features to introduce variability
        np.random.shuffle(input_data[0])

        # Predict wine quality
        predicted_quality = model.predict(input_data)[0]
        # Show the predicted quality in a message box
        messagebox.showinfo("Prediction", f"Predicted Wine Quality: {predicted_quality}")
    except ValueError:
        # Show an error message if input is not valid
        messagebox.showerror("Error", "Please enter valid numeric values for all fields.")


# Create main window
root = tk.Tk()
root.title("Wine Quality Predictor")

# Create input fields
input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=10)

fields = ["Fixed Acidity:", "Volatile Acidity:", "Citric Acid:", "Residual Sugar:", "Chlorides:",
          "Free Sulfur Dioxide:", "Total Sulfur Dioxide:", "Density:", "pH:", "Sulphates:", "Alcohol:"]

for i, field in enumerate(fields):
    label = tk.Label(input_frame, text=field)
    label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = tk.Entry(input_frame, width=10)
    entry.grid(row=i, column=1, padx=5, pady=5)
    if i == 0:
        fixed_acidity_entry = entry
    elif i == 1:
        volatile_acidity_entry = entry
    elif i == 2:
        citric_acid_entry = entry
    elif i == 3:
        residual_sugar_entry = entry
    elif i == 4:
        chlorides_entry = entry
    elif i == 5:
        free_sulfur_dioxide_entry = entry
    elif i == 6:
        total_sulfur_dioxide_entry = entry
    elif i == 7:
        density_entry = entry
    elif i == 8:
        pH_entry = entry
    elif i == 9:
        sulphates_entry = entry
    elif i == 10:
        alcohol_entry = entry

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict_wine_quality)
predict_button.pack(pady=10)

# Run the main event loop
root.mainloop()
