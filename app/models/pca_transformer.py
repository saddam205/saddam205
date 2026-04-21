"""
pca_transformer.py
Part of the app/models module.
"""
"""
PCA feature reduction for noise reduction
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)

class PCATransformer:
    """
    PCA-based feature reduction to remove noise and correlation
    """
    
    def __init__(self, n_components=0.95, use_pca=True):
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, X):
        """
        Fit PCA on training data
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        logger.info(f"Fitting PCA on {X.shape[1]} features...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca.fit(X_scaled)
        self.is_fitted = True
        
        # Log explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"PCA fitted. Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
        logger.info(f"Explained variance ratio: {explained_variance[:5]}")
        
        # Store feature importance
        self.feature_importance = self._calculate_feature_importance()
        
        return self
    
    def transform(self, X):
        """
        Transform features using PCA
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            return self.pca.transform(X_scaled)
        else:
            return X_scaled
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca):
        """
        Inverse transform PCA features back to original space
        """
        X_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_scaled)
    
    def _calculate_feature_importance(self):
        """
        Calculate feature importance from PCA components
        """
        if not self.is_fitted:
            return None
        
        # Sum of absolute loadings across components
        importance = np.abs(self.pca.components_).sum(axis=0)
        importance = importance / importance.sum()
        
        return importance
    
    def get_optimal_components(self, variance_threshold=0.95):
        """
        Find number of components needed for target variance
        """
        if not self.is_fitted:
            return None
        
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        return n_components
    
    def get_feature_importance_ranking(self, feature_names):
        """
        Get ranking of original features by importance
        """
        if self.feature_importance is None:
            return None
        
        ranking = sorted(
            zip(feature_names, self.feature_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranking
    
    def save(self, path_prefix):
        """
        Save PCA transformer components
        """
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(self.pca, f"{path_prefix}_pca.pkl")
        
        # Save metadata
        metadata = {
            'n_components': self.pca.n_components,
            'explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'is_fitted': self.is_fitted
        }
        
        import json
        with open(f"{path_prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"PCA transformer saved to {path_prefix}")
    
    def load(self, path_prefix):
        """
        Load PCA transformer components
        """
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.pca = joblib.load(f"{path_prefix}_pca.pkl")
        self.is_fitted = True
        
        logger.info(f"PCA transformer loaded from {path_prefix}")
        
        return self

# Usage example
def apply_pca_to_trading_features(data, feature_names, n_components=0.95):
    """
    Apply PCA to trading features
    
    Args:
        data: DataFrame with features
        feature_names: List of feature column names
        n_components: Number of components or variance threshold
    
    Returns:
        transformed_features, pca_transformer
    """
    # Extract features
    X = data[feature_names].values
    
    # Create PCA transformer
    pca_transformer = PCATransformer(n_components=n_components)
    
    # Fit and transform
    X_pca = pca_transformer.fit_transform(X)
    
    # Log results
    print(f"Original features: {X.shape[1]}")
    print(f"PCA features: {X_pca.shape[1]}")
    print(f"Variance explained: {pca_transformer.pca.explained_variance_ratio_.sum():.2%}")
    
    # Show feature importance
    importance = pca_transformer.get_feature_importance_ranking(feature_names)
    if importance:
        print("\nTop 10 most important features:")
        for name, imp in importance[:10]:
            print(f"  {name}: {imp:.4f}")
    
    return X_pca, pca_transformer