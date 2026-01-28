import shap
import numpy as np
import tensorflow as tf

class TradeExplainer:
    def __init__(self, model_path):
        # Load your specific model
        self.model = tf.keras.models.load_model(model_path)
        
    def get_explanation(self, input_data, feature_names):
        """
        input_data: The current market features (scaled) used for prediction
        feature_names: List of strings (e.g., ['RSI', 'MACD', 'Close_Price'])
        """
        # SHAP KernelExplainer is versatile for many model types
        # Note: For Deep Learning, we use a small subset of data as a background
        explainer = shap.KernelExplainer(self.model.predict, input_data)
        shap_values = explainer.shap_values(input_data)

        # For the predicted class, find the feature with the highest impact
        # This logic determines "Why Buy" or "Why Sell"
        impacts = shap_values[0][0] 
        max_impact_idx = np.argmax(np.abs(impacts))
        top_feature = feature_names[max_impact_idx]
        
        direction = "positive" if impacts[max_impact_idx] > 0 else "negative"
        
        reason = f"Decision primarily driven by {top_feature} showing a {direction} trend."
        return reason

# Example of how you will call this in your main.py:
# explainer = TradeExplainer('dqn_trading_model.h5')
# reason = explainer.get_explanation(current_market_state, ['Price', 'Volume', 'MA20'])