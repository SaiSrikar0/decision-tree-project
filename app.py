#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

#logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

#folder setup
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, "data", "raw")
clean_dir = os.path.join(base_dir, "data", "cleaned")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

log("application started")
log(f"raw_dir = {raw_dir}")
log(f"clean_dir = {clean_dir}")

#page config
st.set_page_config("End-to-End Decision Tree", layout="wide")
st.title("End-to-End Decision Tree Classifier Application")

#sidebar: model settings
st.sidebar.header("Decision Tree Settings")
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
max_features = st.sidebar.selectbox("Max Features", ["None", "auto", "sqrt", "log2"])

if max_features == "None":
    max_features = None

use_grid_search = st.sidebar.checkbox("Use Grid Search CV")

log(f"Decision Tree settings - criterion: {criterion}, splitter: {splitter}, max_depth: {max_depth}")

#step 1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion started")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])
df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(raw_dir, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.success("Dataset Downloaded successfully")
        log(f"Iris dataset saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(raw_dir, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded file saved at {raw_path}")

#step 2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2: EDA started")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Basic Statistics:**")
        st.write(df.describe())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap")

    log("EDA completed")

#step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    log("Step 3: Data Cleaning started")
    
    strategy = st.selectbox(
        "Missing Value Handling Strategy",
        ["Mean", "Median", "Drop Rows"]
    )
    
    df_clean = df.copy()
    
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
        log("Dropped rows with missing values")
    else:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().sum() > 0:
                if strategy == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    log(f"Filled missing values in {col} with mean")
                elif strategy == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    log(f"Filled missing values in {col} with median")
    
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")
    st.write("**Cleaned Data Preview:**")
    st.dataframe(df_clean.head())
    st.write("**Missing Values After Cleaning:**", df_clean.isnull().sum().sum())
    log("Data cleaning completed")
else:
    st.info("Please complete Step 1 to proceed.")

#step 4: Save cleaned dataset
st.header("Step 4: Save Cleaned Dataset")
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save. Please complete Step 3.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_iris_{timestamp}.csv"
        clean_path = os.path.join(clean_dir, clean_filename)

        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned dataset saved successfully")
        st.info(f"Cleaned dataset saved at {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")
        st.session_state.cleaned_saved = True

#step 5: Load cleaned dataset
st.header("Step 5: Load Cleaned Dataset")
clean_files = os.listdir(clean_dir)
if not clean_files:
    st.warning("No cleaned datasets found. Please save one in Step 4")
    log("No cleaned datasets found")
    df_model = None
else:
    selected = st.selectbox("Select cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(clean_dir, selected))
    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    
    st.dataframe(df_model.head())

#step 6: Train Decision Tree
if df_model is not None:
    st.header("Step 6: Train Decision Tree Classifier")
    log("Step 6: Train Decision Tree started")

    target = st.selectbox("Select target variable", df_model.columns)
    
    if st.button("Train Model"):
        y = df_model[target].copy()
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            log("Target column encoded")
            st.info(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Select numerical features only
        x = df_model.drop(columns=[target])
        x = x.select_dtypes(include=[np.number])
        
        if x.empty:
            st.error("No numerical features available for training.")
            st.stop()

        log(f"Features selected: {list(x.columns)}")
        st.write(f"**Features used:** {list(x.columns)}")
        st.write(f"**Number of samples:** {len(x)}")

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        log(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

        if use_grid_search:
            st.info("Using Grid Search CV for hyperparameter tuning...")
            log("Using Grid Search CV")
            
            param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_depth': [1, 2, 3, 4, 5, 10, 15, 20],
                'max_features': ['auto', 'sqrt', 'log2'],
            }
            
            base_model = DecisionTreeClassifier(random_state=42)
            grid = GridSearchCV(
                base_model, 
                param_grid=param_grid, 
                scoring='accuracy',
                cv=5,
                n_jobs=-1
            )
            
            with st.spinner("Training with Grid Search... This may take a while..."):
                grid.fit(x_train, y_train)
            
            st.success("Grid Search completed!")
            st.write("**Best Parameters:**")
            st.json(grid.best_params_)
            st.write(f"**Best Cross-Validation Score:** {grid.best_score_:.4f}")
            
            model = grid.best_estimator_
            log(f"Best params: {grid.best_params_}")
            log(f"Best CV score: {grid.best_score_:.4f}")
        else:
            # Model initialization with user settings
            model = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
            model.fit(x_train, y_train)
            st.success("Decision Tree model trained successfully")
            log("Decision Tree model trained successfully")

        # Evaluation metrics
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{test_acc:.4f}")
        
        log(f"Train Accuracy: {train_acc:.4f}")
        log(f"Test Accuracy: {test_acc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        log("Confusion matrix displayed")

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        log("Classification report displayed")

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': x.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        log("Feature importance displayed")

        # Decision Tree Visualization
        st.subheader("Decision Tree Visualization")
        
        # Limit tree depth for visualization
        viz_depth = min(3, model.get_depth())
        st.info(f"Showing tree up to depth {viz_depth} for clarity (actual depth: {model.get_depth()})")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model, 
            feature_names=x.columns, 
            filled=True, 
            rounded=True,
            max_depth=viz_depth,
            ax=ax
        )
        st.pyplot(fig)
        log("Decision tree visualization displayed")

        # Model Information
        st.subheader("Model Information")
        st.write(f"**Tree Depth:** {model.get_depth()}")
        st.write(f"**Number of Leaves:** {model.get_n_leaves()}")
        st.write(f"**Number of Features:** {model.n_features_in_}")

        log("Model training and evaluation completed")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit 🎈")
log("Application running")
