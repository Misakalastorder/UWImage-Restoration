import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc, precision_score, recall_score, f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import joblib

# 特征名称
feature_names = [
    "color_bias",
    "color_balance",
    "white_R", "white_G", "white_B",
    "mean_brightness_normalized",
    "kurtosis_normalized",
    "dark_ratio_normalized",
    "lap_var",
    "entropy_value",
    "reblur_value"
]

def train_model(train_csv, k=5):
    # 读取训练数据
    data = pd.read_csv(train_csv)

    X = data[feature_names]
    y = data['Class']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 决策树模型
    clf = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,  # 调整树的最大深度
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=8,
        random_state=41,
        max_leaf_nodes=20,
        min_impurity_decrease=0
    )

    # k折交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=41)
    cv_results = cross_val_score(clf, X_scaled, y, cv=kf, scoring='accuracy')

    print(f"{k}-Fold Cross Validation Accuracy: {cv_results.mean()}")

    # 使用整个数据集训练模型
    clf.fit(X_scaled, y)

    # 模型预测
    y_pred = clf.predict(X_scaled)

    # 计算并输出指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # 决策树可视化
    plt.figure(figsize=(60, 30))  # 增加图像大小
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=clf.classes_, fontsize=4, proportion=True)  # 减小字体大小
    plt.title('Decision Tree')
    plt.savefig('Q1_train_data/decision_tree.eps', format='eps')  # 保存为矢量图
    plt.show()

    # 混淆矩阵可视化
    y_pred = clf.predict(X_scaled)
    conf_mat = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_, annot_kws={"size": 9})
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('Q1_train_data/confusion_matrix.eps', format='eps')  # 保存为矢量图
    plt.show()

    # 保存模型和标准化器
    joblib.dump(clf, 'decision_tree_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

def predict_new_data(features_csv, images_folder,predicted_classes_csv='predicted_classes.csv'):
    # 加载模型和标准化器
    clf = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # 预测新图像数据
    new_data = pd.read_csv(features_csv)
    new_features = new_data[feature_names]
    new_data_scaled = scaler.transform(new_features)
    predictions = clf.predict(new_data_scaled)

    # 保存预测结果
    output = pd.DataFrame({
        'Filename': new_data['image_name'],
        'Predicted Class': predictions
    })
    output.to_csv(predicted_classes_csv, index=False)
    print("Predictions saved")

        # 创建类别文件夹并复制图像
    for class_name in clf.classes_:
        class_folder = os.path.join(images_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    for index, row in output.iterrows():
        src_path = os.path.join(images_folder, row['Filename'])
        dst_path = os.path.join(images_folder, row['Predicted Class'], row['Filename'])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")

if __name__ == '__main__':
    # train_model('train_data.csv', 5)
    #predict_new_data('features.csv', 'Attachment')
    import argparse

    parser = argparse.ArgumentParser(description='Train and predict using Decision Tree model.')
    parser.add_argument('--train', type=str, help='Path to the training CSV file.')
    parser.add_argument('--predict', type=str, help='Path to the features CSV file for prediction.')
    parser.add_argument('--images', type=str, help='Path to the folder containing images for prediction.')
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation.')

    args = parser.parse_args()

    if args.train:
        train_model(args.train, args.k)
    
    if args.predict and args.images:
        predict_new_data(args.predict, args.images)