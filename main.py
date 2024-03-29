import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max.columns", None)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


## Importing Dataset using Pandas Function
data = pd.read_csv(r'https://raw.githubusercontent.com/chandanc5525/BearingFault_Detection/main/without_debris.csv')
df= data.sample(frac=1)    # This command will shuffle the data
df

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats


# Removing outliers using IQR method
def remove_outliers(df):
    Q1 = df[['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC']].quantile(0.25)
    Q3 = df[['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC']].quantile(0.75)
    IQR = Q3 - Q1
    filtered_entries = ~((df[['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC']] < (Q1 - 1.5 * IQR)) | (df[['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC']] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[filtered_entries]

df = remove_outliers(df)

# Splitting data into independent variables (X) and target variable (y)
X = df[['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC']]
y = df['OUTPUT']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining preprocessing steps (scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['RPM', 'X_ACC', 'Y_ACC', 'Z_ACC'])
    ])

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Ensemble': VotingClassifier(estimators=[('rf', RandomForestClassifier(random_state=42)),
                                             ('lr', LogisticRegression(random_state=42)),
                                             ('dt', DecisionTreeClassifier(random_state=42)),
                                             ('knn', KNeighborsClassifier())], voting='soft')
}

# Lists to store metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Train models and evaluate
for name, model in models.items():
    print(f"Evaluating {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics to lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print performance metrics
    print(f"\nAccuracy - {name}: {accuracy:.4f}")
    print(f"Precision - {name}: {precision:.4f}")
    print(f"Recall - {name}: {recall:.4f}")
    print(f"F1 Score - {name}: {f1:.4f}\n")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, ['Non Faulty', 'Faulty'], rotation=45)
    plt.yticks(tick_marks, ['Non Faulty', 'Faulty'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

# Plot bar plots
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(10, 6))
barWidth = 0.15

r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, accuracy_list, color='skyblue', width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, precision_list, color='orange', width=barWidth, edgecolor='grey', label='Precision')
plt.bar(r3, recall_list, color='lightgreen', width=barWidth, edgecolor='grey', label='Recall')
plt.bar(r4, f1_list, color='pink', width=barWidth, edgecolor='grey', label='F1 Score')

# Add labels and legend
plt.xlabel('Model', fontweight='bold')
plt.xticks([r + 1.5*barWidth for r in range(len(models))], models.keys(), rotation=45)
plt.ylabel('Score', fontweight='bold')
plt.title('Performance Metrics for Different Models', fontweight='bold')
plt.legend()
plt.tight_layout()

# Add values on bars
for i in range(len(models)):
    plt.text(i, accuracy_list[i] + 0.01, f"{accuracy_list[i]:.2f}", ha='center', va='bottom')
    plt.text(i + barWidth, precision_list[i] + 0.01, f"{precision_list[i]:.2f}", ha='center', va='bottom')
    plt.text(i + 2*barWidth, recall_list[i] + 0.01, f"{recall_list[i]:.2f}", ha='center', va='bottom')
    plt.text(i + 3*barWidth, f1_list[i] + 0.01, f"{f1_list[i]:.2f}", ha='center', va='bottom')

plt.show()


# Hyper parameter tuning technique
models = {'LogisticRegression':LogisticRegression(),
          'RandomForestClassifier':RandomForestClassifier(),
          'KNNClassifier':KNeighborsClassifier()}

def evaluate(models,X_train,X_test,y_train,y_test):
    np.random.seed(42)
    # Creating One Dictionary to Save Model Score
    model_score = {}
    for name,model in models.items():
        model.fit(X_train, y_train)
        model_score[name] = model.score(X_test,y_test)
    return model_score

model_score = evaluate(models = models ,X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test)
model_score