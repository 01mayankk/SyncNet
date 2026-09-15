# Model Checkpoints Directory — SyncNet

This directory stores serialized machine learning and deep learning model artifacts, encoders, and PyTorch Geometric model weights.

## Saved Model Checkpoints
- `behavioral_logistic_regression.joblib`: Trained Logistic Regression baseline on 12 behavioral profile features.
- `behavioral_random_forest.joblib`: Trained Random Forest Classifier baseline on 12 behavioral profile features.
- `content_logistic_regression.joblib`: Trained Logistic Regression baseline on 384-dim Transformer content embeddings.
- `content_random_forest.joblib`: Trained Random Forest Classifier baseline on 384-dim Transformer content embeddings.

> [!NOTE]
> All production API serving endpoints load saved model checkpoints from this directory during inference.
