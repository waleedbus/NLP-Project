import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib


def build_sample_dataset() -> pd.DataFrame:
 
    data = [
        ("Monthly performance report for department KPIs", "Report"),
        ("Annual financial report and budget summary", "Report"),
        ("Weekly progress report and key achievements", "Report"),
        ("Project status report and risk updates", "Report"),
        ("Employee leave request form submission", "Form"),
        ("New staff onboarding form and details", "Form"),
        ("Travel reimbursement form and receipts", "Form"),
        ("Equipment request form for office supplies", "Form"),
        ("Request for system access approval", "Request"),
        ("Request to update user permissions in the system", "Request"),
        ("Request for meeting room booking and schedule", "Request"),
        ("Request to issue an official letter", "Request"),
        ("Meeting agenda and schedule confirmation", "Email"),
        ("Follow-up email regarding pending approval", "Email"),
        ("Reminder: please submit documents by end of day", "Email"),
        ("Email update about policy changes and guidelines", "Email"),
    ]
    return pd.DataFrame(data, columns=["text", "category"])


def load_data() -> pd.DataFrame:
 
    try:
        df = pd.read_csv("documents.csv")
        required_cols = {"text", "category"}
        if not required_cols.issubset(df.columns):
            raise ValueError("documents.csv must contain columns: text, category")
        return df.dropna(subset=["text", "category"])
    except FileNotFoundError:
        print("No documents.csv found -> using a small built-in sample dataset.")
        return build_sample_dataset()


def main():
    df = load_data()
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.25,
        random_state=42,
        stratify=df["category"] if df["category"].nunique() > 1 else None,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",  
            ngram_range=(1, 2)     
        )),
        ("clf", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.2f}\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "document_classifier.joblib")
    print("\nSaved model to: document_classifier.joblib")

    demo_texts = [
        "Please approve my system access request for the new portal",
        "Attached is the monthly KPI report for operations",
        "Here is the reimbursement form with receipts",
        "Reminder: meeting schedule updated for tomorrow"
    ]
    print("\n=== Demo Predictions ===")
    for t in demo_texts:
        print(f"- Text: {t}")
        print(f"  Predicted category: {model.predict([t])[0]}\n")


if __name__ == "__main__":
    main()
