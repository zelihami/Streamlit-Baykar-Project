import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline


class CancerDatasetApp:
    def __init__(self):
        #Load data on the first run and save it to session state to prevent reloading.
        if "df" not in st.session_state:
            try:
                df = pd.read_csv("data.csv")
                st.session_state.df = df.drop("id",axis=1)
            except FileNotFoundError:
                st.error("ERROR: 'data.csv' not found.")
                st.stop()
        self.df=st.session_state.df

        st.title(" Breast Cancer Dataset Explorer")
        # Start the main application flow.
        self.run()

    def run(self):
        # Create a sidebar for navigating between different pages of the app.
        self.page = st.sidebar.selectbox("Pages", [
            " About Dataset",
            " Statistical Summary",
            " Missing Values",
            " Outlier Analyze",
            " Correlation Analyze",
            " Feature Distribution and PCA",
            " Model"

        ])

        # Simple routing logic to call the function for the selected page.
        if self.page == " About Dataset":
            self.show_about()
        elif self.page == " Statistical Summary":
            self.show_summary()
        elif self.page == " Missing Values":
            self.show_missing()
        elif self.page==" Outlier Analyze":
            self.outlier_analyze()
        elif self.page==" Correlation Analyze":
            self.korelasyon_analiz()
        elif self.page==" Feature Distribution and PCA":
            self.normalization()
        elif self.page==" Model":
            self.model()

    def show_about(self):
        st.markdown("---")
        st.subheader("About the Dataset")
        column_info = pd.DataFrame({
            "Columns": self.df.columns,
            "Data type": self.df.dtypes.values
        })
        st.dataframe(column_info, use_container_width=True)
        object_columns = self.df.select_dtypes(include="object").columns
        object_count = len(object_columns)
        st.markdown(f"Number of obseravtions: {len(self.df)} ")
        st.markdown(f"Number of variables: {len(self.df.columns)}")
        if object_count == 0:
            st.markdown("**There are no columns with `object` data type.**")
        elif object_count == 1:
            col_name = object_columns[0]
            st.markdown(f"**There is just one `object` column: `{col_name}`. Let's look at what's inside.**")
            st.write(f"Unique values in `{col_name}`:")
            st.write(self.df[col_name].value_counts())
            value_counts = self.df[col_name].value_counts()
            labels = value_counts.index
            sizes = value_counts.values
            fig = px.pie(values=sizes, names=labels, title=f"{col_name} distrubition")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(
                f"**There are `{object_count}` columns with `object` data type: {', '.join([f'`{col}`' for col in object_columns])}.**")
            for col in object_columns:
                st.markdown(f"**Unique values in `{col}`:**")
                st.write(self.df[col].value_counts())

        float_cols = [col for col in self.df.select_dtypes(include='float64').columns if col != "Unnamed: 32"]

        #Extract the base features(e.g. radius_mean > radius)
        base_features = sorted(set(col.split("_")[0] for col in float_cols))
        # Get base feature selection from the user

        selected_base = st.selectbox("Select base faeture to visualize:", base_features)
        selected_columns = [col for col in float_cols if col.startswith(selected_base)]
        #Plot graph
        if selected_columns:
            st.markdown(f"### Distribution of features starting with `{selected_base}`")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=self.df[selected_columns], ax=ax, palette="flare")
            ax.set_title(f"Distribution Comparison for '{selected_base.capitalize()}' Features")
            ax.set_ylabel("Value")
            ax.set_xlabel("Features")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No columns found for the selected base feature.")

    def show_summary(self):
        # Display the statistical summary of the dataset.
        st.markdown("---")
        st.subheader("Statistical Summary")
        st.dataframe(self.df.describe().T, use_container_width=True)

    def show_missing(self):
        st.markdown("---")
        st.subheader("Missing Values Overview")
        # Check for and display any missing values.
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            missing_df = missing.reset_index()
            missing_df.columns = ["Column", "Missing Value Count"]
            st.dataframe(missing_df, use_container_width=True)
            # Filter columns with missing values.
            missing_columns = missing_df[missing_df["Missing Value Count"] > 0]["Column"].tolist()

            # If any columns with missing values exist, print them
            if missing_columns:
                st.markdown(f"Columns with missing values: **{', '.join(missing_columns)}**")
            else:
                st.markdown("No missing values found in the dataset.")
            st.markdown("---")
            st.subheader("Remove Missing Values")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Yes, remove missing values"):
                    st.session_state.df=self.df.dropna(axis=1)
                    self.df=st.session_state.df
                    st.success("Missing values have been removed.")


            with col2:
                #Add a fun 'runaway' button for the 'No' option.
                components.html(
                    """
                    <html>
                    <head>
                    <style>
                        #noButton {
                            position: relative;
                            padding: 10px 20px;
                            background-color: green;
                            color: white;
                            border: none;
                            cursor: pointer;
                            font-size: 16px;
                            user-select: none;
                            margin-left: 10px;
                        }
                        #container {
                            display: flex;
                            align-items: center;
                            height: 50px;
                        }
                    </style>
                    </head>
                    <body>
                        <div id="container">
                            <button id="noButton">No</button>
                        </div>
                        <script>
                            const button = document.getElementById("noButton");
                            document.addEventListener("mousemove", function(e) {
                                const mouseX = e.clientX;
                                const mouseY = e.clientY;
                                const rect = button.getBoundingClientRect();

                                const offsetX = rect.left - mouseX;
                                const offsetY = rect.top - mouseY;

                                const distance = Math.sqrt(offsetX ** 2 + offsetY ** 2);

                                if (distance < 80) {
                                    const newX = Math.random() * (window.innerWidth - rect.width);
                                    const newY = Math.random() * (window.innerHeight - rect.height);
                                    button.style.position = "absolute";
                                    button.style.left = newX + "px";
                                    button.style.top = newY + "px";
                                }
                            });
                        </script>
                    </body>
                    </html>
                    """,
                    height=80,
                    scrolling=False
                )

    def outlier_analyze(self):
        self.df = st.session_state.df
        # Helper functions to find and handle outliers using the IQR method.
        def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
            quantile1 = dataframe[col_name].quantile(q1)
            quantile3 = dataframe[col_name].quantile(q3)
            iqr = quantile3 - quantile1
            up_limit = quantile3 + 1.5 * iqr
            low_limit = quantile1 - 1.5 * iqr
            return low_limit, up_limit

        def check_outlier(dataframe, col_name):
            low_limit, up_limit = outlier_thresholds(dataframe, col_name)
            outliers = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]
            return not outliers.empty, outliers.shape[0]

        def replace_with_thresholds(dataframe, variable):
            low_limit, up_limit = outlier_thresholds(dataframe, variable)
            dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
            dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        cols_with_outliers = []
        # Helper functions to find and handle outliers using the IQR method.
        for col in numeric_cols:
            has_outlier, count = check_outlier(self.df, col)
            st.write(f"**{col}**: {'Outlier detected' if has_outlier else 'No outliers'} (count: {count})")
            if has_outlier:
                cols_with_outliers.append(col)

            left, right = st.columns([1, 1])
            with right:
                fig, ax = plt.subplots()
                sns.boxplot(data=self.df[col], ax=ax)
                ax.set_title(f"{col} - Boxplot (Before)")
                st.pyplot(fig)

        if cols_with_outliers:
            if st.button("Apply IQR Thresholds"):
                for col in cols_with_outliers:
                    replace_with_thresholds(self.df, col)
                st.success("The dataset has been adjusted for outliers using the IQR method.")
                st.session_state.df=self.df

                st.write("### Outlier Analyze (After)")

                for col in numeric_cols:
                    has_outlier, count = check_outlier(self.df, col)
                    st.write(f"**{col}**: {'Outlier detected' if has_outlier else 'No outliers'} (count: {count})")

                    left, right = st.columns([1, 1])
                    with right:
                        fig, ax = plt.subplots()
                        sns.boxplot(data=self.df[col], ax=ax)
                        ax.set_title(f"{col} - Boxplot (After)")
                        st.pyplot(fig)

        else:
            st.info("No outliers found in the dataset.")

    def korelasyon_analiz(self):
        # Display the full correlation matrix heatmap.
        self.df=st.session_state.df
        numerical_df = self.df.select_dtypes(include=['number'])
        correlation_matrix = numerical_df.corr()
        st.subheader("Correlation analyze of all features")
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='PiYG', center=0, fmt=".1f", ax=ax)
        st.pyplot(fig)
        df = st.session_state.df.copy()
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Use RFE with a RandomForest to rank feature importance.
        model = RandomForestClassifier(random_state=42).fit(X,y)
        rfe = RFE(estimator=model, n_features_to_select=1)
        rfe.fit(X, y)

        ranking = rfe.ranking_
        feature_ranks = pd.DataFrame({
            'Feature': X.columns,
            'Rank': ranking
        })
        feature_ranks = feature_ranks.sort_values(by="Rank", ascending=True)
        feature_ranks["Importance -inverted rank"] = feature_ranks["Rank"].max() - feature_ranks["Rank"] + 1

        st.subheader("Feature Ranking with RFE")
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Importance (inverted rank)", y="Feature", data=feature_ranks, ax=ax)
        ax.set_title('Feature importance by RFE ranking')
        st.pyplot(fig)

        # Use SelectKBest with f_classif to also rank feature importance.
        st.subheader("Feature importance with SelectKBest")
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)

        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })

        feature_scores = feature_scores.sort_values(by="Score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.barplot(x="Score", y="Feature", data=feature_scores, palette="viridis", ax=ax)
        ax.set_title('Feature importance using SelectKBest (f_classif Score)')
        ax.set_xlabel("ANOVA F-statistic Score")
        ax.set_ylabel("Features")
        st.pyplot(fig)
        # Show correlation matrix for only the top 5 most important features.
        top_5_features_rfe = feature_ranks.head(5)["Feature"].values
        X_top5 = X[top_5_features_rfe]
        correlation_matrix = X_top5.corr()

        st.subheader("Correlation analyze of top 5 features")
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".1f", ax=ax)
        st.pyplot(fig)

    def normalization(self):
        # Display histograms for all numeric features to see their distributions.
        st.header("Distribution and PCA analyze")
        st.info("This page is for analyze only. The data transformations are applied on the 'Model' page to prevent data leakage.")
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        st.subheader("Distribution of Features")
        for i in range(0, len(numeric_cols), 4):
            cols_to_plot = numeric_cols[i:i + 4]
            cols = st.columns(4)
            for j, col in enumerate(cols_to_plot):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(col)
                    st.pyplot(fig)

        st.markdown("---")
        # Perform PCA on a scaled version of the data to find the optimal number of components.
        st.subheader("PCA Analyze to Determine Optimal Components")
        X = df.drop(columns=["diagnosis"], errors='ignore')
        X_numeric = X.select_dtypes(include=np.number)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        # Plot the cumulative explained variance to create an 'elbow' plot.

        pca = PCA()
        pca.fit(X_scaled)
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid(True)
        plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
        plt.legend(loc='best')
        st.pyplot(fig)

    def model(self):
        st.header("Model Training and Evaluation")
        self.df = st.session_state.df
        if 'diagnosis' not in self.df.columns:
            st.error("Target column 'diagnosis' not found in the dataset.")
            return
        le = LabelEncoder()
        y = le.fit_transform(self.df["diagnosis"])
        X = self.df.drop(columns=["diagnosis"])
        st.subheader("1. General Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            classifier_name = st.selectbox(
                "Select Classifier",
                ("SVM", "Random Forest", "XGBoost", "LightGBM", "KNN")
            )
        with col2:
            n_pca_components = st.slider(
                "Number of PCA Components",
                min_value=2, max_value=30, value=10
            )
        with col3:
            test_size = st.slider(
                "Test Set Ratio",
                min_value=0.1, max_value=0.5, value=0.25
            )

        st.subheader("2. Hyperparameter Tuning Method")
        use_grid_search = st.checkbox(
            "Find Best Parameters with GridSearchCV",
            value=False,
            help="If unchecked, you can set hyperparameters manually below."
        )

        params = {}
        if not use_grid_search:
            with st.expander(f"Manual Hyperparameter Tuning for {classifier_name}", expanded=True):
                if classifier_name == 'SVM':
                    C = st.slider('C', 0.1, 15.0, 1.0)
                    kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly'])
                    params = {'classifier__C': C, 'classifier__kernel': kernel}
                elif classifier_name == 'Random Forest':
                    n_estimators = st.slider('n_estimators (RF)', 50, 500, 100)
                    max_depth = st.slider('max_depth (RF)', 2, 20, 5)
                    params = {'classifier__n_estimators': n_estimators, 'classifier__max_depth': max_depth}
                elif classifier_name == 'XGBoost':
                    n_estimators = st.slider('n_estimators (XGB)', 50, 500, 100)
                    max_depth = st.slider('max_depth (XGB)', 2, 15, 5)
                    learning_rate = st.slider('learning_rate (XGB)', 0.01, 0.5, 0.1)
                    params = {'classifier__n_estimators': n_estimators, 'classifier__max_depth': max_depth,
                              'classifier__learning_rate': learning_rate}
                elif classifier_name == 'LightGBM':
                    n_estimators = st.slider('n_estimators (LGBM)', 50, 500, 100)
                    max_depth = st.slider('max_depth (LGBM)', 2, 15, 5)
                    learning_rate = st.slider('learning_rate (LGBM)', 0.01, 0.5, 0.1)
                    params = {'classifier__n_estimators': n_estimators, 'classifier__max_depth': max_depth,
                              'classifier__learning_rate': learning_rate}
                elif classifier_name == 'KNN':
                    n_neighbors = st.slider('K (Neighbors)', 1, 50, 5)
                    weights = st.selectbox('Weights', ['uniform', 'distance'])
                    params = {'classifier__n_neighbors': n_neighbors, 'classifier__weights': weights}

        classifiers = {
            'SVM': SVC(probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }

        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('classifier', classifiers[classifier_name])
        ])

        #Model training
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            with st.spinner("Model training in progress..."):
                if use_grid_search:
                    param_grids = {
                        'SVM': {
                            'classifier__C': [0.1, 1, 10, 50],
                            'classifier__kernel': ['rbf', 'linear']
                        },
                        'Random Forest': {
                            'classifier__n_estimators': [100, 200, 300],
                            'classifier__max_depth': [5, 10, 20, None]
                        },
                        'XGBoost': {
                            'classifier__n_estimators': [100, 200],
                            'classifier__max_depth': [3, 5, 7],
                            'classifier__learning_rate': [0.05, 0.1, 0.2]
                        },
                        'LightGBM': {
                            'classifier__n_estimators': [100, 200],
                            'classifier__max_depth': [3, 5, 7],
                            'classifier__learning_rate': [0.05, 0.1, 0.2]
                        },
                        'KNN': {
                            'classifier__n_neighbors': [3, 5, 7, 9, 11],
                            'classifier__weights': ['uniform', 'distance']
                        }
                    }
                    search = GridSearchCV(final_pipeline, param_grids[classifier_name], cv=5, n_jobs=-1,
                                          scoring='accuracy')
                    search.fit(X_train, y_train)
                    st.success("GridSearchCV complete!")
                    st.info(f"Best parameters found: `{search.best_params_}`")
                    model_to_evaluate = search.best_estimator_
                else:
                    final_pipeline.set_params(**params)
                    final_pipeline.fit(X_train, y_train)
                    model_to_evaluate = final_pipeline

            st.success("Model training complete!")
            st.subheader("Model Evaluation Results")
            y_pred = model_to_evaluate.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="Model Accuracy", value=f"{acc:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Classification Report**")
                st.text(classification_report(y_test, y_pred, target_names=le.classes_))
            with col2:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)



if __name__ == "__main__":
    CancerDatasetApp()


