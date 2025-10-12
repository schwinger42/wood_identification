"""
Traditional Machine Learning models for wood identification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import joblib


class WoodClassifier:
    """
    Wrapper class for traditional ML classifiers
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize classifier
        
        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'gradient_boosting', 'mlp')
            **kwargs: Additional parameters for the classifier
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """
        Create the specified model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        
        elif self.model_type == 'svm':
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42),
                probability=True
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                activation=kwargs.get('activation', 'relu'),
                solver=kwargs.get('solver', 'adam'),
                alpha=kwargs.get('alpha', 0.0001),
                batch_size=kwargs.get('batch_size', 'auto'),
                learning_rate=kwargs.get('learning_rate', 'constant'),
                max_iter=kwargs.get('max_iter', 200),
                random_state=kwargs.get('random_state', 42)
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Train the classifier
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        
        Returns:
            self
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Data of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Data of shape (n_samples, n_features)
        
        Returns:
            Class probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support probability prediction")
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X: Data of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        return self.model.score(X, y)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            cv: Number of cross-validation folds
        
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        self.model = joblib.load(filepath)
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models)
        
        Returns:
            Feature importance array
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise AttributeError(f"{self.model_type} does not support feature importance")


class EnsembleClassifier:
    """
    Ensemble of multiple classifiers
    """
    
    def __init__(self, classifiers):
        """
        Initialize ensemble
        
        Args:
            classifiers: List of WoodClassifier instances
        """
        self.classifiers = classifiers
    
    def fit(self, X, y):
        """
        Train all classifiers
        
        Args:
            X: Training data
            y: Training labels
        
        Returns:
            self
        """
        for clf in self.classifiers:
            clf.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions using voting
        
        Args:
            X: Data to predict
        
        Returns:
            Predicted labels
        """
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        # Majority voting
        from scipy import stats
        final_predictions = stats.mode(predictions, axis=0)[0].flatten()
        return final_predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities using averaging
        
        Args:
            X: Data to predict
        
        Returns:
            Average class probabilities
        """
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])
        return np.mean(probas, axis=0)
