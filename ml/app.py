import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  VotingClassifier
from imblearn.over_sampling import SMOTE
import shap
import math
# Set random seed for reproducibility
np.random.seed(42)


# 1. Data Loading and Exploration
def load_and_explore_data(filepath='wdbc.data'):
    print("Loading dataset...")
    columns = [
        "ID", "Class",
        "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean", "Compactness_Mean",
        "Concavity_Mean", "ConcavePoints_Mean", "Symmetry_Mean", "FractalDimension_Mean",
        "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE", "Compactness_SE",
        "Concavity_SE", "ConcavePoints_SE", "Symmetry_SE", "FractalDimension_SE",
        "Radius_Worst", "Texture_Worst", "Perimeter_Worst", "Area_Worst", "Smoothness_Worst",
        "Compactness_Worst", "Concavity_Worst", "ConcavePoints_Worst", "Symmetry_Worst", "FractalDimension_Worst"
    ]
    df = pd.read_csv(filepath,  header=None, names=columns)

    df['Class'] = df['Class'].map({'M': 1, 'B': 0})

    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")

    print("\nChecking for missing values:")
    print(df.isnull().sum().any())


    print("\nBasic statistics for anonymized features:")
    print(df.describe())

    return df


# 2. Data Preprocessing
def preprocess_data(df):
    print("\nPreprocessing data...")

    #drop the ID column and label
    X = df.drop('Class', axis=1).drop('ID',axis=1)
    y = df['Class']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalanced data with SMOTE (only for training data)
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"After SMOTE - Training set shape: {X_train_resampled.shape}")
    print(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler, X_train.columns




# 4. SVM Model Implementation
def train_svm_models(X_train, y_train):
    print("\nTraining SVM models with different kernels...")


    svm_models = {
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Polynomial)': SVC(kernel='poly', degree=3, probability=True, random_state=42)
    }


    for name, model in svm_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    return svm_models


# 5. Ensemble Methods Implementation
def train_ensemble_models(X_train, y_train, svm_models):
    print("\nTraining ensemble models...")

    num_models = 20
    sample_size = 200
    feature_sample_size = 30
    np.random.seed(0)

    sample_weights = np.ones(X_train.shape[0]) / X_train.shape[0]

    all_models_bagging = []
    all_models_rf = []
    all_models_boosting = []
    alphas = []

    # for normal bagging using decision tree
    for m in range(num_models):
        sample_idx = np.random.choice(X_train.shape[0], sample_size)
        X_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]
        model1 = DecisionTreeClassifier()
        model1.fit(X_train_sample, y_train_sample)
        all_models_bagging.append(model1)

    # for bagging using Random Forest
    for m in range(num_models):
        sample_idx = np.random.choice(X_train.shape[0], sample_size)
        X_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]
        model2 = DecisionTreeClassifier(max_features=feature_sample_size)
        model2.fit(X_train_sample, y_train_sample)
        all_models_rf.append(model2)

    # For boosting
    for m in range(num_models):
        sample_idx = np.random.choice(X_train.shape[0], sample_size)
        X_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]

        model = DecisionTreeClassifier(max_depth=10)
        model.fit(X_train_sample, y_train_sample, sample_weights[sample_idx])

        error = 1 - model.score(X_train_sample, y_train_sample, sample_weights[sample_idx])

        if error > 0:
            alpha = np.log((1 - error) / error)
        else:
            alpha = 1  # Assign strong confidence for perfect classifiers
        alphas.append(alpha)

        incorrect_predictions = model.predict(X_train_sample) != y_train_sample
        sample_weights[sample_idx] *= np.exp(alpha * incorrect_predictions)

        all_models_boosting.append(model)




    # Create voting ensemble based on svm model
    estimators = [
        ('svm_linear', svm_models['SVM (Linear)']),
        ('svm_poly', svm_models['SVM (Polynomial)']),
        ('svm_rbf', svm_models['SVM (RBF)'])
    ]

    voting_clf = VotingClassifier(estimators=estimators, voting='soft')

    print("Training Voting Ensemble...")
    voting_clf.fit(X_train, y_train)


    all_models = {
        'SVM (Linear)': svm_models['SVM (Linear)'],
        'SVM (RBF)': svm_models['SVM (RBF)'],
        'SVM (Polynomial)': svm_models['SVM (Polynomial)'],
        'Voting Ensemble': voting_clf
    }
    ensemble_models = {
        'bagging': all_models_bagging,
        'boosting': (all_models_boosting, alphas),
        'Random Forest': all_models_rf,
    }

    return all_models, ensemble_models

def bagging_predict(test_data, all_models):
    votes = np.zeros((test_data.shape[0], len(all_models)))
    combined_predictions = np.zeros(test_data.shape[0])

    for idx, m in enumerate(all_models):
        votes[:, idx] = m.predict(test_data)

    for test_point in range(votes.shape[0]):
        combined_predictions[test_point] = np.bincount(np.int64(votes[test_point])).argmax()

    return combined_predictions


def boosting_predict(test_data, all_models_3, alphas):
    votes2 = np.zeros((test_data.shape[0], len(all_models_3)))
    combined_predictions = np.zeros(test_data.shape[0])

    for idx, m in enumerate(all_models_3):
        votes2[:, idx] = m.predict(test_data)

    for test_point in range(len(votes2)):
        combined_predictions[test_point] = np.bincount(np.int64(votes2[test_point]), alphas).argmax()

    return combined_predictions

# 6. Model Evaluation
def evaluate_models(models, ensemble_models, X_test, y_test):
    print("\nEvaluating models...")

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}:")

        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        
        print(classification_report(y_test, y_pred))

        
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        results[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'confusion_matrix': cm,
        }
    for name, model in ensemble_models.items():
        print(f"\nEvaluating {name}:")
        if name == 'boosting':
            all_models = model[0]
            alphas = model[1]
            y_pred =  boosting_predict(X_test, all_models, alphas)
        else:
            y_pred =bagging_predict(X_test, model)

        print(classification_report(y_test, y_pred))


        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        results[name] = {
            'y_pred': y_pred,
            'confusion_matrix': cm,
        }

    return results


# 7. Visualization Functions
def plot_evaluation_metrics(results):
    print("\nPlotting evaluation metrics...")

    
    num_models = len(results)


    cols = min(3, num_models)  
    rows = math.ceil(num_models / cols)  

    # Adjust figure size dynamically
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)

    # Flatten axes array for easy iteration
    axes = np.array(axes).reshape(-1)

    # Iterate and plot each confusion matrix
    for ax, (name, result) in zip(axes, results.items()):
        cm = result['confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)

        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('Actual Label', fontsize=10)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # Hide empty subplots
    for i in range(num_models, len(axes)):
        fig.delaxes(axes[i])

    # Save the image with optimal spacing
    plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")

    plt.show()

# 9. Visualization of Explanations
def visualize_explanations_Shap(all_models, ensemble_models, X_train, X_test, feature_names):
    # select the models to visualize
    # selected_models = {
    #     'Bagging (last model)': ensemble_models['bagging'][19],  
    #     'Random Forest (last model)': ensemble_models['Random Forest'][19], 
    #     'Boosting (last model)': ensemble_models['boosting'][0][19],  
    # }

    model =  ensemble_models['boosting'][0][19]
   
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values.feature_names = feature_names

    np.shape(shap_values.values)
    print("SHAP values shape:", np.shape(shap_values.values))

    # waterfall plot can be applied only on single samples
    shap.plots.waterfall(shap_values[55, :, 1])

    # this is beeswarm applied on all samples of label 0 Benign
    shap.summary_plot(shap_values[:, :, 0], X_test)

    # this is beeswarm plot applied on all samples of label 1 Malignant
    shap.summary_plot(shap_values[:, :, 1], X_test)

    # this is bar plot applied on all samples of label 1 Malignant
    shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar")

    print("\nAll SHAP plots generated and saved!")

# 10. Main Function
def main():
   
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    svm_models = train_svm_models(X_train, y_train)
    all_models, ensemble_models = train_ensemble_models(X_train, y_train, svm_models)
    evaluation_results = evaluate_models(all_models, ensemble_models, X_test, y_test)
    plot_evaluation_metrics(evaluation_results)
    visualize_explanations_Shap(all_models, ensemble_models, X_train, X_test, feature_names)
    print("\nExecution completed! Check the output directory for evaluation metrics and explanations.")


if __name__ == "__main__":
    main()