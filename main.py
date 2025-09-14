from pathlib import Path
from src.data.data_loader import DataLoader
from src.models.model import SentimentClassifier

def main():
        """Main function to run the training and evaluation pipeline."""
        print("--- Starting Sentiment Analysis Project ---")

        # 1. Define paths
        project_root = Path(__file__).parent
        data_path = project_root / "data" / "reviews.csv"
        model_path = project_root / "models" / "sentiment_model.joblib"

        # 2. Load and split data
        print("\n--- Loading Data ---")
        data_loader = DataLoader(data_path)
        X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
        print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples.")

        # 3. Train the model
        print("\n--- Training Model ---")
        classifier = SentimentClassifier()
        classifier.train(X_train, y_train)
        print("Model training complete.")

        # 4. Evaluate the model
        print("\n--- Evaluating Model ---")
        classifier.evaluate(X_test, y_test)

        # 5. Save the model
        classifier.save_model(model_path)

        # 6. Test with a sample prediction
        print("\n--- Sample Prediction ---")
        sample_text = ["هذا التطبيق ممتاز وسهل الاستخدام"]
        prediction = classifier.predict(sample_text)
        print(f"Prediction for '{sample_text[0]}': {prediction[0]}")

        print("\n--- Project Execution Finished ---")

if __name__ == "__main__":
        main()