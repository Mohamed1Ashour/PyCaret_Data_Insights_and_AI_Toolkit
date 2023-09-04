import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, make_scorer, mean_squared_error
import joblib
import warnings
from math import sqrt


class data_insights_toolkit:
    def __init__(self, data):
        self.df = data
        self.categorical_data = self.df.select_dtypes(
            include=['object']).columns
        self.numerical_data = self.df.select_dtypes(
            include=['int', 'float']).columns

    def data_description(self):
        st.subheader("Data Overview")
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#333;">   </h3>',
                 unsafe_allow_html=True)

        st.dataframe(self.df.head(5))

        st.subheader("Data Types Overview")
        data_types_df = pd.DataFrame(
            {'Column Name': self.df.columns, 'Data Type': self.df.dtypes})
        st.table(data_types_df)

        st.subheader("Columns Names")
        st.table(pd.DataFrame({'Column Name': self.df.columns}))

        st.subheader("Columns with Missing Values")
        missing_values = self.df.isna().sum().reset_index()
        missing_values.columns = ["Column Name", "Missing Values"]
        st.table(missing_values)
        for col in self.categorical_data:
            st.subheader(f"Column: '{col}'")
            unique_categories = self.df[col].nunique()
            missing_percentage = (
                self.df[col].isna().sum() / len(self.df)) * 100
            st.write(f"- Number of Unique Categories: {unique_categories}")
            st.write(
                f"- Percentage of Missing Values: {missing_percentage:.2f}%")

    def statistics(self):
        st.subheader("Statistics")
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#FFFFFF;">Correlation</h3>',
                 unsafe_allow_html=True)
        st.dataframe(self.df.corr())
        st.write('</div>', unsafe_allow_html=True)
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#FFFFFF;">Coefficient of Variation</h3>',
                 unsafe_allow_html=True)
        for col in self.df.select_dtypes(include=np.number).corr():
            cv = (self.df[col].std() / self.df[col].mean()) * 100
            st.write("-" * 50)
            st.write(f"Column: {col}")
            st.write(f"Coefficient of Variation: {cv:.2f}")
        st.write('</div>', unsafe_allow_html=True)
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#FFFFFF;">Descriptive Statistics</h3>',
                 unsafe_allow_html=True)
        for col in self.numerical_data:
            st.write("-" * 50)
            st.subheader(f"Descriptive Statistics for '{col}':")
            st.dataframe(self.df[col].describe())
            st.write(f"Skewness: {self.df[col].skew()}")
            st.write(f"Kurtosis: {self.df[col].kurt()}")
            st.write("-" * 50)
        st.write('</div>', unsafe_allow_html=True)

    def percentage_of_outliers_in_columns(self):
        outlier_percentages = {}
        for col in self.numerical_data:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower_bound)
                               | (self.df[col] > upper_bound)]
            outlier_percentage = round((len(outliers) / len(self.df)) * 100)
            outlier_percentages[col] = outlier_percentage
        outlier_df = pd.DataFrame(list(outlier_percentages.items()), columns=[
                                  "Column Name", "Outlier Percentage"])
        st.subheader("Percentage of Outliers in Columns")
        st.table(outlier_df)

    def download_data(self, data, filename):
        if st.button("Download Data"):
            with open(filename, 'rb') as file:
                data_bytes = file.read()
            st.download_button(label="Click to Download",
                               data=data_bytes, file_name=filename)

    def handling_missing_values(self):
        for col in self.numerical_data:
            missing_percentage = (
                self.df[col].isna().sum() / len(self.df)) * 100
            if missing_percentage <= 5.0:
                self.df.dropna(subset=[col], inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        filtered_data_filename = "data_without_missing_Values.csv"
        self.df.to_csv(filtered_data_filename, index=False)

        st.subheader("Download Filtered Data")
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#333;">   </h3>',
                 unsafe_allow_html=True)

        self.download_data(self.df, filtered_data_filename)

    def remove_outliers(self):
        for col in self.numerical_data:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            self.df = self.df[(self.df[col] >= lower_bound)
                              & (self.df[col] <= upper_bound)]

        filtered_data_filename = "filtered_data.csv"
        self.df.to_csv(filtered_data_filename, index=False)

        st.subheader("Download Data Without Outliers")
        st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                 unsafe_allow_html=True)
        st.write('<h3 style="color:#333;">   </h3>',
                 unsafe_allow_html=True)

        self.download_data(self.df, filtered_data_filename)

    def display_results(self, name):
        st.subheader(f"Data After {name}")
        self.data_description()
        self.statistics()

    def apply_get_dummies(self, column_name):
        dummies = pd.get_dummies(
            self.df[column_name], prefix=column_name, drop_first=False)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[column_name], inplace=True)

        st.write(
            f"Applied pd.get_dummies to column '{column_name}' and dropped the first dummy variable.")
        st.dataframe(self.df)


#   -------------------------------------------------------------------------------------- Data Visulization ------------------------------------------------------------------

    def plot_histograms_plots(self, x):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.df[x], kde=True, ax=ax)
        ax.set_title(f'Histogram of {x}')
        ax.set_xlabel(x)
        ax.set_ylabel('Frequency')

        st.pyplot(fig)

    def plot_kde_plots(self, x):
        fig, ax = plt.subplots()
        sns.kdeplot(self.df[x], shade=True, ax=ax)
        plt.title(f'KDE Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Density')
        st.pyplot(fig)

    def plot_ecdf_plots(self, x):
        fig, ax = plt.subplots()
        sns.ecdfplot(data=self.df, x=x, ax=ax)
        plt.title(f'ECDF Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Cumulative Probability')
        st.pyplot(fig)

    def plot_regplots(self, x, y):
        fig, ax = plt.subplots()
        sns.regplot(data=self.df, x=x, y=y, ax=ax)
        plt.title(f'Regression Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot(fig)

    def plot_scatter_plots(self, x, y):
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df, x=x, y=y, ax=ax)
        plt.title(f'Scatter Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot(fig)

    def plot_line_plots(self, x, y):
        fig, ax = plt.subplots()
        sns.lineplot(data=self.df, x=x, y=y, ax=ax)
        plt.title(f'Line Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot(fig)

    def plot_box_plots(self, x):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.df, y=x, ax=ax)
        plt.title(f'Box Plot of {x}')
        plt.ylabel(x)
        st.pyplot(fig)

    def plot_count_plots(self, x):
        fig, ax = plt.subplots()
        sns.countplot(data=self.df, x=x, ax=ax)
        plt.title(f'Count Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    def plot_bar_plots(self, x, y):
        fig, ax = plt.subplots()
        sns.barplot(data=self.df, x=x, y=y, ax=ax)
        plt.title(f'Bar Plot of {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    def plot_point_plots(self, x, y):
        fig, ax = plt.subplots()
        sns.pointplot(data=self.df, x=x, y=y, ax=ax)
        plt.title(f'Point Plot of {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=90)
        st.pyplot(fig)

#   -------------------------------------------------------------------------------------- Machine Learning ------------------------------------------------------------------

    def evaluate_regression(self, y_column):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            X = self.df.drop(columns=[y_column])
            y = self.df[y_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_accuracies = {}
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(),
                'SVR': SVR(),
                'Decision Tree Regression': DecisionTreeRegressor(),
                'Random Forest Regression': RandomForestRegressor(),
                'KNN Regression': KNeighborsRegressor()
            }
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                rmse = sqrt(mean_squared_error(y_test, y_pred))
                model_accuracies[model_name] = rmse

            model_accuracies_df = pd.DataFrame(
                {'Model': model_accuracies.keys(), 'RMSE': model_accuracies.values()})

            st.markdown("### Model Performance")
            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.write("Here are the RMSE values for different regression models:")
            st.table(model_accuracies_df)

            best_model = min(model_accuracies,
                             key=lambda x: model_accuracies[x])
            best_rmse = model_accuracies[best_model]

            st.markdown("### Best Model")

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.write(
                f"The best model is **{best_model}** with RMSE: **{best_rmse:.2f}**")

            model_filename = "best_regression_model.joblib"
            joblib.dump(models[best_model], model_filename)

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Download Model")

            if st.button(f"Download Best Model ({best_model})"):
                with open(model_filename, 'rb') as model_file:
                    model_bytes = model_file.read()
                st.download_button(label="Click to Download", data=model_bytes,
                                   file_name=model_filename, key=f"{best_model}_download")

        except ValueError as ve:
            st.error(
                f"ValueError: {ve}. Make sure all columns '{list(self.df.columns)}' are numerical for regression or preprocess the data.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")

    def evaluate_regression_with_cv(self, y_column, num_folds=5):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            X = self.df.drop(columns=[y_column])
            y = self.df[y_column]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model_accuracies = {}

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(),
                'SVR': SVR(),
                'Decision Tree Regression': DecisionTreeRegressor(),
                'Random Forest Regression': RandomForestRegressor(),
                'KNN Regression': KNeighborsRegressor()
            }

            for model_name, model in models.items():
                cv_scores = cross_val_score(
                    model, X_scaled, y, cv=num_folds, scoring='neg_root_mean_squared_error')
                rmse_scores = -cv_scores
                rmse_mean = np.mean(rmse_scores)
                model_accuracies[model_name] = rmse_mean

            model_accuracies_df = pd.DataFrame(
                {'Model': model_accuracies.keys(), 'RMSE': model_accuracies.values()})

            st.markdown("### Model Performance")
            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.write("Here are the RMSE values for different regression models:")
            st.table(model_accuracies_df)

            best_model = min(model_accuracies,
                             key=lambda x: model_accuracies[x])
            best_rmse = model_accuracies[best_model]

            st.markdown("### Best Model")

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.write(
                f"The best model is **{best_model}** with RMSE: **{best_rmse:.2f}**")

            model_filename = "best_regression_cv_model.joblib"
            joblib.dump(models[best_model], model_filename)

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Download Model")

            if st.button(f"Download Best Regression CV Model ({best_model})"):
                with open(model_filename, 'rb') as model_file:
                    model_bytes = model_file.read()
                st.download_button(label="Click to Download", data=model_bytes,
                                   file_name=model_filename, key=f"{best_model}_cv_download")

        except ValueError as ve:
            st.error(
                f"ValueError: {ve}. Make sure all columns '{list(self.df.columns)}' are numerical for regression or preprocess the data.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")

    def evaluate_classification_models(self, y_column):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            X = self.df.drop(columns=[y_column])
            y = self.df[y_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'k-NN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }

            model_results = []

            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                model_results.append({
                    'Model': model_name,
                    'Accuracy': round(accuracy, 2),
                    'Precision': round(precision, 2),
                    'Recall': round(recall, 2),
                    'F1 Score': round(f1, 2)
                })

            st.subheader(
                "Classification Model Evaluation with Cross-Validation")

            result_df = pd.DataFrame(model_results)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.write(result_df)

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Best Model")
            best_model = result_df.loc[result_df['F1 Score'].idxmax()]
            st.write(best_model)

            best_model_name = best_model['Model']
            best_model_filename = f"best_classification_model_{best_model_name}.joblib"
            joblib.dump(models[best_model_name], best_model_filename)

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Download Model")

            if st.button(f"Download Best Classification Model ({best_model_name})"):
                with open(best_model_filename, 'rb') as model_file:
                    model_bytes = model_file.read()
                st.download_button(label="Click to Download", data=model_bytes,
                                   file_name=best_model_filename, key=f"{best_model_name}_download")

        except Exception as e:
            st.error(
                f"An error occurred: {e}. Please check your data and model selection.")

    def evaluate_classification_models_with_cv(self, y_column, num_folds=5, shuffle=True):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            X = self.df.drop(columns=[y_column])
            y = self.df[y_column]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'k-NN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }

            model_results = []

            for model_name, model in models.items():
                cv_accuracy_scores = cross_val_score(
                    model, X_scaled, y, cv=num_folds, scoring='accuracy')
                accuracy_mean = np.mean(cv_accuracy_scores)
                precision_scorer = make_scorer(
                    precision_score, average='weighted')
                cv_precision_scores = cross_val_score(
                    model, X_scaled, y, cv=num_folds, scoring=precision_scorer)
                precision_mean = np.mean(cv_precision_scores)
                recall_scorer = make_scorer(recall_score, average='weighted')
                cv_recall_scores = cross_val_score(
                    model, X_scaled, y, cv=num_folds, scoring=recall_scorer)
                recall_mean = np.mean(cv_recall_scores)
                f1_scorer = make_scorer(f1_score, average='weighted')
                cv_f1_scores = cross_val_score(
                    model, X_scaled, y, cv=num_folds, scoring=f1_scorer)
                f1_mean = np.mean(cv_f1_scores)

                model_results.append({
                    'Model': model_name,
                    'Accuracy': round(accuracy_mean, 2),
                    'Precision': round(precision_mean, 2),
                    'Recall': round(recall_mean, 2),
                    'F1 Score': round(f1_mean, 2)
                })

            st.subheader(
                "Classification Model Evaluation with Cross-Validation")
            result_df = pd.DataFrame(model_results)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)
            st.write(result_df)
            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Best Model")
            best_model = result_df.loc[result_df['F1 Score'].idxmax()]
            st.write(best_model)

            best_model_name = best_model['Model']
            best_model_filename = f"best_classification_cv_model_{best_model_name}.joblib"
            joblib.dump(models[best_model_name], best_model_filename)

            st.write('<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">',
                     unsafe_allow_html=True)
            st.write('<h3 style="color:#333;">   </h3>',
                     unsafe_allow_html=True)

            st.subheader("Download Model")

            if st.button(f"Download Best Classification CV Model ({best_model_name})"):
                with open(best_model_filename, 'rb') as model_file:
                    model_bytes = model_file.read()
                st.download_button(label="Click to Download", data=model_bytes,
                                   file_name=best_model_filename, key=f"{best_model_name}_download")

        except Exception as e:
            st.error(
                f"An error occurred: {e}. Please check your data and model selection.")


def main():

    st.title("PyCaret- Data Insights and AI Toolkit")

    file_path = st.sidebar.file_uploader(
        "Upload a file", type=["csv", "xls", "xlsx", "dta"])
    if file_path is not None:
        if file_path.type == "application/vnd.ms-excel":
            df = pd.read_excel(file_path)
        elif file_path.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file_path)
        elif file_path.type == "text/csv":
            df = pd.read_csv(file_path)
        elif file_path.type == "application/x-stata":
            df = pd.read_stata(file_path)

        analyzer = data_insights_toolkit(df)

        st.sidebar.subheader("Select a function:")

        st.sidebar.write(
            '<div style="background-color:#F4F4F4; padding: 10px; border-radius: 5px;">', unsafe_allow_html=True)

        selected_function = st.sidebar.radio("Navigator", [
                                             "Data Description", "Statistics", "Outliers and Missing Values", "Data Transformation", "Plots", "Model Evaluation", "Quit"])

        if selected_function == "Quit":
            sys.exit(0)

        elif selected_function == "Data Description":
            st.write(analyzer.data_description())

        elif selected_function == "Statistics":
            st.write(analyzer.statistics())

        elif selected_function == "Outliers and Missing Values":
            sub_option = st.sidebar.checkbox("Show Percentage of Outliers")
            if sub_option:
                st.write(analyzer.percentage_of_outliers_in_columns())

            sub_option = st.sidebar.checkbox("Handling Missing Values")
            if sub_option:
                st.write(analyzer.handling_missing_values())
                st.write(analyzer.display_results('Handling Missing Values'))

            sub_option = st.sidebar.checkbox("Remove Outliers")
            if sub_option:
                st.write(analyzer.remove_outliers())
                st.write(analyzer.display_results('Remove Outliers'))

        elif selected_function == "Data Transformation":
            sub_option = st.sidebar.checkbox("Apply get dummies")
            if sub_option:
                available_columns = analyzer.df.columns.tolist()
                column_name = st.selectbox(
                    "Select a column for pd.get_dummies:", available_columns)
                if st.button("Apply get dummies"):
                    analyzer.apply_get_dummies(column_name)

        elif selected_function == "Plots":
            sub_option = st.sidebar.radio("Select Plot Type", [
                                          "Histograms", "KDE Plots", "ECDF Plots", "Regression Plots", "Scatter Plots", "Line Plots", "Box Plots", "Count Plots", "Bar Plots", "Point Plots"])

            if sub_option == "Histograms":
                selected_column = st.selectbox(
                    "Select a numerical column for histogram:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_histograms_plots(selected_column)

            elif sub_option == "KDE Plots":
                selected_column = st.selectbox(
                    "Select a numerical column for KDE plot:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_kde_plots(selected_column)

            elif sub_option == "ECDF Plots":
                selected_column = st.selectbox(
                    "Select a numerical column for ECDF plot:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_ecdf_plots(selected_column)

            elif sub_option == "Regression Plots":
                x = st.selectbox("Select the X-axis (independent variable):",
                                 df.select_dtypes(include=['number']).columns)
                y = st.selectbox("Select the Y-axis (dependent variable):",
                                 df.select_dtypes(include=['number']).columns)
                analyzer.plot_regplots(x, y)

            elif sub_option == "Scatter Plots":
                x = st.selectbox("Select x-axis column:",
                                 df.select_dtypes(include=['number']).columns)
                y = st.selectbox("Select y-axis column:",
                                 df.select_dtypes(include=['number']).columns)
                analyzer.plot_scatter_plots(x, y)

            elif sub_option == "Line Plots":
                x = st.selectbox(
                    "Select x-axis column:", df.select_dtypes(include=['number']).columns)
                y = st.selectbox(
                    "Select y-axis column:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_line_plots(x, y)

            elif sub_option == "Box Plots":
                selected_column = st.selectbox(
                    "Select a numerical column for box plot:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_box_plots(selected_column)

            elif sub_option == "Count Plots":
                selected_column = st.selectbox(
                    "Select a categorical column for count plot:", df.select_dtypes(exclude=['number']).columns)
                analyzer.plot_count_plots(selected_column)

            elif sub_option == "Bar Plots":
                selected_x_column = st.selectbox(
                    "Select a categorical column for x-axis:", df.select_dtypes(exclude=['number']).columns)
                selected_y_column = st.selectbox(
                    "Select a numerical column for y-axis:",  df.select_dtypes(include=['number']).columns)
                analyzer.plot_bar_plots(selected_x_column, selected_y_column)

            elif sub_option == "Point Plots":
                selected_x_column = st.selectbox(
                    "Select a categorical column for x-axis:", df.select_dtypes(exclude=['number']).columns)
                selected_y_column = st.selectbox(
                    "Select a numerical column for y-axis:", df.select_dtypes(include=['number']).columns)
                analyzer.plot_point_plots(selected_x_column, selected_y_column)

        elif selected_function == "Model Evaluation":
            sub_option = st.sidebar.radio(
                "Select Model Type", ["Regression", "Classification"])

            if sub_option == "Regression":
                selected_y_column = st.selectbox(
                    "Select the target numerical column for regression:", df.select_dtypes(include=['number']).columns)
                sub_option = st.sidebar.checkbox(
                    "Evaluate with Cross-Validation")
                if sub_option:
                    num_folds = st.slider(
                        "Select the number of cross-validation folds (k):", min_value=2, max_value=10, value=5)
                    analyzer.evaluate_regression_with_cv(
                        selected_y_column, num_folds)
                else:
                    analyzer.evaluate_regression(selected_y_column)

            elif sub_option == "Classification":
                selected_y_column = st.selectbox(
                    "Select the target categorical column for classification:", df.columns)
                sub_option = st.sidebar.checkbox(
                    "Evaluate with Cross-Validation")
                if sub_option:
                    num_folds = st.slider(
                        "Select the number of cross-validation folds (k):", min_value=2, max_value=10, value=5)
                    analyzer.evaluate_classification_models_with_cv(
                        selected_y_column, num_folds)
                else:
                    analyzer.evaluate_classification_models(selected_y_column)


if __name__ == "__main__":
    main()
