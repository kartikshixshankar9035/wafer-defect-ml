import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import joblib

from pandas.compat import pickle_compat

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

sys.modules["pandas.indexes"] = pd.core.indexes

print("Loading WM-811K Dataset...")

with open("dataset/LSWMD.pkl", "rb") as file:
    df = pickle_compat.Unpickler(
        file, encoding="latin1"
    ).load()

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# 2. EXTRACT FAILURE LABELS
# ==========================================

def extract_label(x):
    try:
        return str(x[0][0])
    except (IndexError, TypeError):
        return "unknown"


df["failure_label"] = df["failureType"].apply(extract_label)

print("\nFailure Type Distribution:")
print(df["failure_label"].value_counts())

# ==========================================
# 3. PREPARE DATASET
# ==========================================

df_labeled = df[
    df["failure_label"] != "unknown"
].copy()

print("\nLabelled Dataset Shape:", df_labeled.shape)

print("\nClass Distribution:")
print(df_labeled["failure_label"].value_counts())

os.makedirs("results", exist_ok=True)

# ==========================================
# 4. VISUALIZE CLASS DISTRIBUTION
# ==========================================

class_counts = df_labeled["failure_label"].value_counts()

plt.figure(figsize=(10, 5))
class_counts.plot(kind="bar")

plt.title("Wafer Defect Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/class_distribution.png", dpi=300)
plt.close()

# ==========================================
# 5. VISUALIZE WAFER DEFECT PATTERNS
# ==========================================

classes = [
    "none", "Center", "Donut",
    "Edge-Loc", "Edge-Ring", "Loc",
    "Random", "Scratch", "Near-full"
]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for ax, label in zip(axes.flat, classes):

    sample = df_labeled[
        df_labeled["failure_label"] == label
    ].iloc[0]

    ax.imshow(sample["waferMap"], cmap="viridis")
    ax.set_title(label)
    ax.axis("off")

fig.suptitle("WM-811K Wafer Defect Patterns")
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("results/wafer_defect_patterns.png", dpi=300)
plt.close()

# ==========================================
# 6. SAMPLE WAFER ANALYSIS
# ==========================================

sample_map = df_labeled.iloc[0]["waferMap"]

total_dies = np.count_nonzero(sample_map)
good_dies = np.count_nonzero(sample_map == 1)
defective_dies = np.count_nonzero(sample_map == 2)

defect_percentage = (
    defective_dies / total_dies * 100
    if total_dies else 0
)

print("\n================================")
print("SAMPLE WAFER ANALYSIS")
print("================================")

print("Wafer Map Shape:", sample_map.shape)
print("Unique Values:", np.unique(sample_map))
print("Total Dies:", total_dies)
print("Good Dies:", good_dies)
print("Defective Dies:", defective_dies)
print("Defect Percentage:", round(defect_percentage, 2), "%")

# ==========================================
# 7. FEATURE EXTRACTION - 3x3 GRID
# ==========================================

def extract_features(wafer):

    wafer = np.array(wafer)

    total_dies = np.count_nonzero(wafer)
    good_dies = np.count_nonzero(wafer == 1)
    defective_dies = np.count_nonzero(wafer == 2)

    defect_ratio = (
        defective_dies / total_dies
        if total_dies else 0
    )

    features = [
        good_dies,
        defective_dies,
        defect_ratio
    ]

    row_parts = np.array_split(wafer, 3, axis=0)

    for row in row_parts:

        col_parts = np.array_split(row, 3, axis=1)

        for region in col_parts:

            defect_count = np.count_nonzero(region == 2)
            features.append(defect_count)

    return features


print("\nExtracting Features from All Wafer Maps...")

X = np.array([
    extract_features(w)
    for w in df_labeled["waferMap"]
])

y = df_labeled["failure_label"].values

print("\nFeature Extraction Completed!")
print("Feature Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Split Completed!")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# 9. RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# 10. MODEL EVALUATION
# ==========================================

print("\nEvaluating Model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1-Score:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(xticks_rotation=45, cmap="Blues")

plt.title("Wafer Defect Classification")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion Matrix Saved!")

# ==========================================
# 12. FEATURE IMPORTANCE
# ==========================================

feature_names = [
    "Good Dies",
    "Defective Dies",
    "Defect Ratio",
    "Top Left",
    "Top Center",
    "Top Right",
    "Middle Left",
    "Middle Center",
    "Middle Right",
    "Bottom Left",
    "Bottom Center",
    "Bottom Right"
]

importances = model.feature_importances_

plt.figure(figsize=(10, 6))

plt.barh(feature_names, importances)

plt.xlabel("Importance Score")
plt.title("Random Forest - Feature Importance")

plt.tight_layout()

plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

print("\nFeature Importance Analysis Completed!")

# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

joblib.dump(
    model,
    "results/wafer_random_forest_best.pkl"
)

print("\nBest Model Saved Successfully!")

# ==========================================
# 14. FINAL RESULTS
# ==========================================

print("\n================================")
print("PROJECT EXECUTION COMPLETED")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1-Score:", round(macro_f1, 4))

print("\nAll visualizations saved in results folder.")