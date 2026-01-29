import joblib

MODEL_PATH = "document_classifier.joblib"

def main():
    model = joblib.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
    print("Type a document text, or 'exit' to quit.\n")

    while True:
        text = input("Document text> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            print("Please enter some text.\n")
            continue

        pred = model.predict([text])[0]
        proba = None
        # Some models support predict_proba; MultinomialNB does
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            labels = model.classes_
            proba = dict(zip(labels, probs))

        print(f"Predicted category: {pred}")
        if proba:
            top = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top probabilities:", ", ".join([f"{k}:{v:.2f}" for k, v in top]))
        print()

if __name__ == "__main__":
    main()
