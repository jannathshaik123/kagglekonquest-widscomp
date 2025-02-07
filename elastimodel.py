from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,GridSearchCV, cross_validate,cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, RobustScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV,ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor


# Define paths
BASE_PATH = "widsdatathon2025-university"
TRAIN_PATH = os.path.join(BASE_PATH, "train_tsv/train_tsv")
TEST_PATH = os.path.join(BASE_PATH, "test_tsv/test_tsv")
METADATA_PATH = os.path.join(BASE_PATH, "metadata")
DATA_PATH = "data"  # New folder for processed data

# Create data directory if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)

# def extract_subject_id(filename):
#     """
#     Extract subject ID from the complex filename format
#     Example: sub-NDARAA075AMK_ses-HBNsiteSI_task-rest_acq-VARIANTObliquity_atlas-Schaefer2018p200n17_space-MNI152NLin6ASym_reg-36Parameter_desc-PearsonNilearn_correlations.tsv
#     """
#     # Extract the subject ID (everything between 'sub-' and the first '_')
#     subject_id = filename.split('sub-')[1].split('_')[0]
#     return subject_id

def preprocess_data(data, encoders=None, is_training=True):
    """
    Comprehensive preprocessing including categorical variables
    """
    if encoders is None:
        encoders = {}
    
    # Handle categorical columns
    categorical_cols = ['sex', 'study_site', 'ethnicity', 'race', 'handedness', 
                       'parent_1_education', 'parent_2_education']
    
    for col in categorical_cols:
        if col in data.columns:
            if is_training:
                # Create new encoder for training data
                encoders[col] = LabelEncoder()
                data[col] = encoders[col].fit_transform(data[col].fillna('missing'))
            else:
                # Use existing encoder for test data
                data[col] = encoders[col].transform(data[col].fillna('missing'))
    
    # Handle numerical missing values
    numerical_cols = ['bmi', 'p_factor_fs', 'internalizing_fs', 'externalizing_fs', 'attention_fs']
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mean())
    
    return data, encoders

# def load_and_process_connectome(file_path):
#     """Load and process a single connectome matrix"""
#     matrix = pd.read_csv(file_path, sep='\t', header=None)
#     upper_tri = matrix.values[np.triu_indices(200, k=1)]
#     return upper_tri

# def process_dataset(folder_path, metadata_path=None):
#     """Process dataset and save to file"""
#     features = []
#     subject_ids = []
    
#     print("Processing connectome matrices...")
#     for file in os.listdir(folder_path):
#         if file.endswith('.tsv'):
#             subject_id = extract_subject_id(file)
#             file_path = os.path.join(folder_path, file)
#             try:
#                 connectome_features = load_and_process_connectome(file_path)
#                 features.append(connectome_features)
#                 subject_ids.append(subject_id)
#             except Exception as e:
#                 print(f"Error processing {file}: {str(e)}")
    
#     # Create DataFrame
#     feature_names = [f'feature_{i}' for i in range(len(features[0]))]
#     df = pd.DataFrame(features, columns=feature_names)
#     df['participant_id'] = subject_ids
    
#     # Merge with metadata if provided
#     if metadata_path:
#         metadata = pd.read_csv(metadata_path)
#         df = pd.merge(df, metadata, on='participant_id', how='left')
    
#     return df


def enhance_connectome_features(features_df):
    """Enhanced feature engineering for connectome data"""
    # Create copy to avoid modifying original
    df = features_df.copy()
    
    # Add network-level features
    for i in range(0, 19900, 200):  # Assuming 200x200 matrix
        region_features = df.iloc[:, i:i+200]
        df[f'region_{i//200}_mean'] = region_features.mean(axis=1)
        df[f'region_{i//200}_std'] = region_features.std(axis=1)
        df[f'region_{i//200}_max'] = region_features.max(axis=1)
    
    # Add global network features
    df['global_mean'] = df.filter(like='feature_').mean(axis=1)
    df['global_std'] = df.filter(like='feature_').std(axis=1)
    df['global_connectivity'] = df.filter(like='feature_').sum(axis=1)
    
    return df

def create_optimized_elastic_pipeline():
    """Create an optimized pipeline for ElasticNet with feature processing"""
    pipeline = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson')),
        ('pca', PCA(n_components=0.95)),
        ('elasticnet', ElasticNet(random_state=42))
    ])
    
    param_grid = {
        'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
        'elasticnet__l1_ratio': [0.01, 0.03, 0.05, 0.1, 0.15],
        'elasticnet__tol': [1e-4],
        'elasticnet__max_iter': [10000]
    }
    
    return pipeline, param_grid

def create_stacked_model():
    """Create a stacked model combining ElasticNet with other regressors"""
    estimators = [
        ('elasticnet1', ElasticNet(alpha=0.01, l1_ratio=0.03)),
        ('elasticnet2', ElasticNet(alpha=0.1, l1_ratio=0.05)),
        ('elasticnet3', ElasticNet(alpha=0.05, l1_ratio=0.01))
    ]
    
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=ElasticNet(alpha=0.01, l1_ratio=0.03),
        cv=5
    )
    
    return stacked_model

def evaluate_model_cv(model, X, y, cv_splits=5, random_state=42):
    """Evaluate model using cross-validation with multiple metrics"""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    # Perform cross-validation
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Process scores
    results = {
        'test_rmse': -scores['test_rmse'],
        'train_rmse': -scores['train_rmse'],
        'test_mae': -scores['test_mae'],
        'train_mae': -scores['train_mae'],
        'test_r2': scores['test_r2'],
        'train_r2': scores['train_r2']
    }
    
    # Calculate statistics
    stats = {metric: {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    } for metric, scores in results.items()}
    
    return stats

def train_and_evaluate_models(X, y, n_splits=5, random_state=42):
    """Train both single and stacked models with cross-validation"""
    # Enhance features
    X_enhanced = enhance_connectome_features(X)
    
    # Create and train models
    pipeline, param_grid = create_optimized_elastic_pipeline()
    stacked_model = create_stacked_model()
    
    # GridSearch for single model with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=n_splits,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_enhanced, y)
    
    # Get best single model
    best_single_model = grid_search.best_estimator_
    
    # Train stacked model
    stacked_model.fit(X_enhanced, y)
    
    # Evaluate both models with cross-validation
    single_model_stats = evaluate_model_cv(best_single_model, X_enhanced, y, 
                                         cv_splits=n_splits, random_state=random_state)
    stacked_model_stats = evaluate_model_cv(stacked_model, X_enhanced, y, 
                                          cv_splits=n_splits, random_state=random_state)
    
    return {
        'single_model': {
            'model': best_single_model,
            'best_params': grid_search.best_params_,
            'cv_stats': single_model_stats
        },
        'stacked_model': {
            'model': stacked_model,
            'cv_stats': stacked_model_stats
        }
    }

def print_cv_results(results):
    """Print formatted cross-validation results"""
    for model_name, model_info in results.items():
        print(f"\n{model_name.replace('_', ' ').title()} Results:")
        stats = model_info['cv_stats']
        
        print("\nRMSE:")
        print(f"Test - Mean: {stats['test_rmse']['mean']:.3f} ± {stats['test_rmse']['std']:.3f}")
        print(f"Train - Mean: {stats['train_rmse']['mean']:.3f} ± {stats['train_rmse']['std']:.3f}")
        
        print("\nMAE:")
        print(f"Test - Mean: {stats['test_mae']['mean']:.3f} ± {stats['test_mae']['std']:.3f}")
        print(f"Train - Mean: {stats['train_mae']['mean']:.3f} ± {stats['train_mae']['std']:.3f}")
        
        print("\nR²:")
        print(f"Test - Mean: {stats['test_r2']['mean']:.3f} ± {stats['test_r2']['std']:.3f}")
        print(f"Train - Mean: {stats['train_r2']['mean']:.3f} ± {stats['train_r2']['std']:.3f}")
        
        if 'best_params' in model_info:
            print("\nBest Parameters:", model_info['best_params'])

# Example usage:
"""
# Assuming you have your data split into X_train, X_test, y_train, y_test
models = train_elastic_models(X_train, y_train, X_test, y_test)

# Print results for each model
for name, model_info in models.items():
    print(f"\n{name}:")
    print(f"Best RMSE: {model_info['best_score']:.3f}")
    print("Best parameters:", model_info['best_params'])
"""

if __name__ == "__main__":
    # #TRAIN DATA
    # Process training data
    # print("Processing training data...")
    # train_metadata_path = os.path.join(METADATA_PATH, "training_metadata.csv")
    # train_data = process_dataset(TRAIN_PATH, train_metadata_path)
    
    # # Save processed data
    # train_data.to_csv(os.path.join(DATA_PATH, 'processed_train_data.csv'), index=False)
    # print(f"Processed data saved to {DATA_PATH}")
    
    # # #TEST DATA
    # # Process testing data
    # test_metadata_path = os.path.join(METADATA_PATH, "test_metadata.csv")
    # test_data = process_dataset(TEST_PATH, test_metadata_path)
    
    # # Save processed data
    # test_data.to_csv(os.path.join(DATA_PATH, 'processed_test_data.csv'), index=False)
    # print(f"Processed data saved to {DATA_PATH}")
    
    train_data = pd.read_csv(os.path.join(DATA_PATH, 'processed_train_data.csv'))
    print(train_data.head(), train_data.shape)
    # Prepare data for modeling
    feature_cols = [col for col in train_data.columns if col not in ['participant_id', 'age']]
    X = train_data[feature_cols]
    y = train_data['age']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_enc, encoders = preprocess_data(X, is_training=True)
    X_val, _ = preprocess_data(X_val, encoders=encoders, is_training=False)
    
    # Train and evaluate multiple models
    results = train_and_evaluate_models(X_enc, y, n_splits=5, random_state=42)
    
    # Print detailed results
    print_cv_results(results)

    # Save the model with better average RMSE
    if results['single_model']['cv_stats']['test_rmse']['mean'] < results['stacked_model']['cv_stats']['test_rmse']['mean']:
        best_model = results['single_model']['model']
    else:
        best_model = results['stacked_model']['model']
        
print("Model training complete!")