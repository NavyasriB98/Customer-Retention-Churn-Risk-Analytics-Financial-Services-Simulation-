import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_data(n_samples=5000):
    print(f"Generating synthetic data with {n_samples} records...")
    
    data = {
        'CustomerID': [f'CUST-{i:05d}' for i in range(n_samples)],
        'Gender': np.random.choice(['Male', 'Female', 'female', 'M'], n_samples), # Intentionally messy
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'Tenure': np.random.randint(0, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No', 'Fiber'], n_samples), # messy
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73]) # Typical churn rate
    }
    
    df = pd.DataFrame(data)
    
    # Simulate TotalCharges (correlated with Tenure and MonthlyCharges) + some noise
    df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure'] + np.random.normal(0, 10, n_samples)
    df['TotalCharges'] = df['TotalCharges'].abs().round(2) # Ensure positive
    
    # Introduce missing values to simulate real-world dirty data
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, 'TotalCharges'] = np.nan
    
    print("Data generation complete.")
    return df

def clean_data(df):
    print("\n--- Starting Data Cleaning ---")
    
    # 1. Handle Missing Values
    initial_missing = df['TotalCharges'].isnull().sum()
    print(f"Missing TotalCharges before cleaning: {initial_missing}")
    
    # Impute missing TotalCharges with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("Filled missing values with median.")
    
    # 2. Standardize Categorical Values
    print("Standardizing 'Gender' and 'InternetService' columns...")
    df['Gender'] = df['Gender'].replace({'female': 'Female', 'M': 'Male'})
    df['InternetService'] = df['InternetService'].replace({'Fiber': 'Fiber optic'})
    
    # 3. Ensure distinct values are correct
    print("Unique Genders:", df['Gender'].unique())
    print("Unique InternetService:", df['InternetService'].unique())
    
    return df

def feature_engineering(df):
    print("\n--- Feature Engineering ---")
    
    # Create simpler categories or numeric conversions if needed
    # For this demo, we'll keep it simple for Power BI readability
    
    # Binning Tenure (Cohorts)
    labels = ["0-12 Months", "12-24 Months", "24-48 Months", "Over 48 Months"]
    bins = [0, 12, 24, 48, 100]
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=bins, labels=labels, right=False)
    
    print("Added 'TenureGroup' column.")
    return df

def build_model(df):
    print("\n--- Building Predictive Model ---")
    
    # Prepare data for modeling
    model_df = df.copy()
    
    # Drop ID columns that aren't predictors
    X = model_df.drop(['CustomerID', 'Churn', 'TenureGroup'], axis=1)
    y = model_df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # --- Feature Importance ---
    importances = rf.feature_importances_
    features = X.columns
    print("\nTop 5 Important Features:")
    indices = np.argsort(importances)[::-1]
    for i in range(5):
        print(f"{i+1}. {features[indices[i]]} ({importances[indices[i]]:.4f})")
        
    return rf, le_dict, X.columns

def score_full_dataset(df, model, le_dict, feature_cols):
    print("\n--- Scoring Full Dataset ---")
    
    # Prepare data for prediction (must match training format)
    score_df = df.copy()
    X_score = score_df.drop(['CustomerID', 'Churn', 'TenureGroup'], axis=1)
    
    for col in X_score.select_dtypes(include=['object']).columns:
        if col in le_dict:
            le = le_dict[col]
            # Handle unknown labels in production by assigning a default or mode (simplification here)
            # For synthesis, we know labels match.
            X_score[col] = le.transform(X_score[col])
            
    # Ensure column order matches training
    X_score = X_score[feature_cols]
    
    # Predict Probability
    probs = model.predict_proba(X_score)[:, 1] # Probability of 'Yes' class
    preds = model.predict(X_score)
    
    # Add to original dataframe
    df['Churn_Probability'] = probs.round(4)
    df['Predicted_Churn_Label'] = ['Yes' if p > 0.5 else 'No' for p in probs]
    
    # Create Risk Categories
    conditions = [
        (df['Churn_Probability'] < 0.3),
        (df['Churn_Probability'] >= 0.3) & (df['Churn_Probability'] < 0.7),
        (df['Churn_Probability'] >= 0.7)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    df['Risk_Category'] = np.select(conditions, choices, default='Medium Risk')
    
    return df

def main():
    # 1. Generate
    df = generate_data(5000)
    
    # 2. Clean
    df_clean = clean_data(df)
    
    # 3. Feature Engineering (for Analysis)
    df_fe = feature_engineering(df_clean)
    
    # 4. Model
    model, le_dict, feature_cols = build_model(df_fe)
    
    # 5. Score
    df_final = score_full_dataset(df_fe, model, le_dict, feature_cols)
    
    # 6. Export
    output_path = 'churn_modeling_output.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Exported cleaned and scored data to '{output_path}'")

if __name__ == "__main__":
    main()
