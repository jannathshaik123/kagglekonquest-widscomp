from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Create a pipeline that includes scaling and ElasticNet
def create_elastic_pipeline(scaler_type='standard'):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('elasticnet', ElasticNet(random_state=42))
    ])
    
    # Define parameter grid with more granular values
    param_grid = {
        'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'elasticnet__l1_ratio': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        'elasticnet__tol': [1e-4, 1e-3],
        'elasticnet__max_iter': [10000]
    }
    
    return pipeline, param_grid

# Function to train and evaluate models
def train_elastic_models(X_train, y_train, X_test, y_test):
    models = {}
    
    # Try both scaling approaches
    for scaler_type in ['standard', 'robust']:
        pipeline, param_grid = create_elastic_pipeline(scaler_type)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store the results
        models[f'elasticnet_{scaler_type}'] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_  # Convert back to positive RMSE
        }
    
    return models

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