import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import random

# 1. Score mapping for dropdowns
score_map = {
    "age": {
        "18 to 25": 1,
        "26 to 35": 2,
        "36 to 45": 3,
        "46 plus": 4
    },
    "gender": {
        "male": 2,
        "female": 2,
        "others": 1
    },
    "marital_status": {
        "single": 1,
        "married": 3,
        "divorced": 2,
        "widowed": 2
    },
    "employment_type": {
        "unemployed": 1,
        "private": 2,
        "government": 3,
        "self-employment": 4
    },
    "monthly_income": {
        "less than $50,000": 1,
        "$50,000 - $100,000": 2,
        "$100,000 - $250,000": 3,
        "$250,000 plus": 4
    },
    "credit_history": {
        "poor": 1,
        "average": 2,
        "good": 3
    },
    "education_level": {
        "high school": 1,
        "undergraduate": 2,
        "graduate": 3,
        "post-graduate": 4
    },
    "work_experience": {
        "less than 1 year": 1,
        "1 to 3 years": 2,
        "4 to 6 years": 3,
        "7 years plus": 4
    }
}

# 2. Generate synthetic dataset (coded variables)
def generate_dataset(n=200):
    data = []
    for _ in range(n):
        entry = {}
        total_score = 0

        for field, options in score_map.items():
            selected_option = random.choice(list(options.keys()))
            score = score_map[field][selected_option]
            entry[field] = score
            total_score += score

        # Calculate loan: (score * 20000) + 100
        entry["loan_amount"] = (total_score * 20000) + 100
        data.append(entry)

    df = pd.DataFrame(data)
    return df

# 3. Train and save model
def train_and_save_model():
    df = generate_dataset()

    X = df.drop("loan_amount", axis=1)
    y = df["loan_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model to disk
    joblib.dump(model, "loan_model.pkl")
    print("✅ Model trained and saved as loan_model.pkl")

# 4. Load and predict with model
def predict(input_dict):
    model = joblib.load("loan_model.pkl")
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    return prediction[0]

# 5. Example usage
if __name__ == "__main__":
    train_and_save_model()

    # Test prediction
    sample_input = {
        "age": 4,  # e.g. "46 plus"
        "gender": 2,
        "marital_status": 3,
        "employment_type": 4,
        "monthly_income": 4,
        "credit_history": 3,
        "education_level": 4,
        "work_experience": 4
    }

    loan = predict(sample_input)
    print(f"💰 Predicted Loan Amount: ${loan:,.2f}")