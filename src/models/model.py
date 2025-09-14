"""
Sentiment Analysis Model
========================

This module contains the SentimentClassifier class for Arabic text sentiment analysis.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import numpy as np

class SentimentClassifier:
    """
    A sentiment analysis classifier for Arabic text using TF-IDF and Naive Bayes.
    
    This class provides a complete pipeline for:
    - Text preprocessing and feature extraction using TF-IDF
    - Classification using Multinomial Naive Bayes
    - Model training, evaluation, and persistence
    
    Attributes:
        pipeline: sklearn Pipeline containing TF-IDF vectorizer and classifier
    """
    
    def __init__(self):
        """
        Initialize the sentiment classifier with a TF-IDF + Naive Bayes pipeline.
        """
        print("🔧 إنشاء نموذج تحليل المشاعر...")
        
        # إنشاء pipeline متكامل
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                # معلمات TF-IDF
                ngram_range=(1, 2),      # استخدام كلمات مفردة وثنائية
                max_features=5000,       # أقصى عدد ميزات
                lowercase=True,          # تحويل إلى أحرف صغيرة
                stop_words=None,         # عدم إزالة كلمات الوقف (مهمة للعربية)
                min_df=1,                # الحد الأدنى لتكرار الكلمة
                max_df=0.95              # الحد الأقصى لتكرار الكلمة
            )),
            ('classifier', MultinomialNB(
                alpha=1.0                # معلمة التنعيم
            ))
        ])
        
        print("✅ تم إنشاء النموذج بنجاح!")

    def train(self, X_train, y_train):
        """
        Train the sentiment classifier on the provided data.
        
        Args:
            X_train: Training text data (pandas Series or list)
            y_train: Training labels (pandas Series or list)
            
        Returns:
            self: Returns the trained classifier instance for method chaining
        """
        print("🚀 بدء تدريب النموذج...")
        print(f"📊 عدد عينات التدريب: {len(X_train)}")
        
        # عرض توزيع الفئات
        if hasattr(y_train, 'value_counts'):
            print("📋 توزيع الفئات:")
            for label, count in y_train.value_counts().items():
                print(f"   {label}: {count} عينة")
        
        # تدريب النموذج
        self.pipeline.fit(X_train, y_train)
        
        print("✅ تم تدريب النموذج بنجاح!")
        return self

    def predict(self, X):
        """
        Make predictions on new text data.
        
        Args:
            X: Text data to predict (list, pandas Series, or single string)
            
        Returns:
            numpy.ndarray: Predicted sentiment labels
        """
        # التحقق من تدريب النموذج
        if not hasattr(self.pipeline.named_steps['classifier'], 'classes_'):
            raise ValueError("❌ النموذج غير مدرب! استخدم train() أولاً.")
        
        # التعامل مع النص المفرد
        if isinstance(X, str):
            X = [X]
            
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities for each class.
        
        Args:
            X: Text data to predict (list, pandas Series, or single string)
            
        Returns:
            numpy.ndarray: Prediction probabilities for each class
        """
        # التحقق من تدريب النموذج
        if not hasattr(self.pipeline.named_steps['classifier'], 'classes_'):
            raise ValueError("❌ النموذج غير مدرب! استخدم train() أولاً.")
        
        # التعامل مع النص المفرد
        if isinstance(X, str):
            X = [X]
            
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        
        Args:
            X_test: Test text data
            y_test: True test labels
            
        Returns:
            dict: Classification report as dictionary
        """
        print("📊 تقييم أداء النموذج...")
        
        # التنبؤ على بيانات الاختبار
        y_pred = self.predict(X_test)
        
        # تقرير التصنيف
        print("\n" + "="*60)
        print("📈 تقرير التصنيف التفصيلي")
        print("="*60)
        
        # الحصول على التقرير كقاموس
        report = classification_report(
            y_test, 
            y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        # طباعة التقرير المنسق
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # طباعة الدقة الإجمالية
        accuracy = report['accuracy']
        print(f"🎯 الدقة الإجمالية: {accuracy:.2%}")
        
        # مصفوفة الخلط
        print(f"\n📊 مصفوفة الخلط:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return report

    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model (string or Path object)
        """
        path = Path(path)
        
        # إنشاء المجلد إذا لم يكن موجوداً
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # حفظ النموذج
        joblib.dump(self.pipeline, path)
        print(f"💾 تم حفظ النموذج في: {path}")
    
    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model (string or Path object)
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"❌ ملف النموذج غير موجود: {path}")
        
        # تحميل النموذج
        self.pipeline = joblib.load(path)
        print(f"📂 تم تحميل النموذج من: {path}")
    
    def get_feature_names(self):
        """
        Get the feature names from the TF-IDF vectorizer.
        
        Returns:
            numpy.ndarray: Array of feature names
        """
        if not hasattr(self.pipeline.named_steps['tfidf'], 'vocabulary_'):
            raise ValueError("❌ النموذج غير مدرب! استخدم train() أولاً.")
        
        return self.pipeline.named_steps['tfidf'].get_feature_names_out()
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        if not hasattr(self.pipeline.named_steps['classifier'], 'classes_'):
            return {"status": "غير مدرب"}
        
        info = {
            "status": "مدرب",
            "classes": list(self.pipeline.named_steps['classifier'].classes_),
            "n_features": len(self.get_feature_names()),
            "vectorizer_params": self.pipeline.named_steps['tfidf'].get_params(),
            "classifier_params": self.pipeline.named_steps['classifier'].get_params()
        }
        
        return info
