import joblib
import pandas as pd
import argparse
from pathlib import Path

PROJECT_ROOT = Path.cwd()
MODELS = PROJECT_ROOT / "models"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict Titanic survival")
parser.add_argument("--pclass", type=int, required=True, help="Passenger class (1, 2, or 3)")
parser.add_argument("--sex", type=str, required=True, help="Sex (male or female)")
parser.add_argument("--age", type=float, required=True, help="Age in years")
parser.add_argument("--sibsp", type=int, default=0, help="Number of siblings/spouses")
parser.add_argument("--parch", type=int, default=0, help="Number of parents/children")
parser.add_argument("--fare", type=float, required=True, help="Ticket fare")
parser.add_argument("--embarked", type=str, default="S", help="Port (S, C, or Q)")

args = parser.parse_args()

# Load the model
model = joblib.load(MODELS / "titanic_survival.joblib")

# Create DataFrame with input
passenger = pd.DataFrame([{
    "Pclass": args.pclass,
    "Sex": args.sex,
    "Age": args.age,
    "SibSp": args.sibsp,
    "Parch": args.parch,
    "Fare": args.fare,
    "Embarked": args.embarked,
}])

# Predict
survival_prob = model.predict_proba(passenger)[:, 1][0]
survival = model.predict(passenger)[0]

print(f"Passenger: {args.sex}, age {args.age}, class {args.pclass}, fare ${args.fare}")
print(f"Survival probability: {survival_prob:.1%}")
print(f"Prediction: {'Survived' if survival == 1 else 'Did not survive'}")
