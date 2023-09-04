import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import keyboard
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, make_scorer, mean_squared_error
import warnings
from math import sqrt


class Pycharet:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        if self.file_path.endswith('.csv'):
            self.df = pd.read_csv(self.file_path)

        elif (self.file_path.endswith('.xls')) or (self.file_path.endswith('.xlsx')):
            self.df = pd.read_excel(self.file_path)

        elif self.file_path.endwith('.dta'):
            self.df = pd.read_stata(self.file_path)

        self.categorical_data = self.df.select_dtypes(
            include=['object']).columns
        self.numerical_data = self.df.select_dtypes(
            include=['int', 'float']).columns

    def data_description(self):
        print("Data Overview:")
        print(self.df.head(5))
        print('-' * 50)
        print('\n')

        print("Data Types Overview:")
        print(self.df.dtypes)
        print("-" * 50)
        print('\n')

        print('Columns Names')
        print(self.df.columns)
        print('-' * 50)
        print('\n')

        print("Columns with Missing Values:")
        print(self.df.isna().sum())
        print("-" * 50)
        print('\n')

        for col in self.categorical_data:
            print(f"'{col}':")
            print("Number of Unique Categories:", self.df[col].nunique())
            print("Percentage of Missing Values:",
                  (self.df[col].isna().sum()/len(self.df)) * 100)
            print('\n')

    def statistics(self):

        print("Correlation")
        print(self.df.corr())
        print("-" * 50)
        print('\n')

        for col in self.df.select_dtypes(include=np.number).corr():
            print(f"Column: {col}")
            print(
                f"Coefficient of Variation =  {(self.df[col].std() / self.df[col].mean()) * 100:.2f}")
            print("-" * 50)
            print('\n')

        for col in self.numerical_data:
            print(f"Descriptive Statistics for '{col}':")
            print(self.df[col].describe())
            print('\n')

            print(f"Skewness: {self.df[col].skew()}")
            print(f"Kurtosis: {self.df[col].kurt()}")
            print("-" * 50)
            print('\n')

    def percentage_of_outliers_in_columns(self):
        outlier_percentages = {}

        for col in self.numerical_data:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 - 1.5*iqr
            outliers = self.df[(self.df[col] < lower_bound)
                               | (self.df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(self.df)) * 100
            outlier_percentages[col] = outlier_percentage

        for key, value in outlier_percentages.items():
            print("\n")
            print(f"{key} ==> {value}")
            print("-" * 50)

    def handling_missing_values(self):
        for col in self.numerical_data:
            missing_percentage = (
                self.df[col].isna().sum() / len(self.df)) * 100
            if missing_percentage <= 5.0:
                self.df.dropna(subset=[col], inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        print("-" * 50)
        print("Your Data After Handling Missing Values ==> ")
        print("-" * 50)
        self.data_description()
        print("-" * 50)
        self.statistics()

    def remove_outliers(self):
        for col in self.numerical_data:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 - 1.5*iqr
            self.df = self.df[(self.df[col] >= lower_bound)
                              & (self.df[col] <= upper_bound)]
        print("-" * 50)
        print("Your Data After Handling Missing Values ==> ")
        print("-" * 50)
        self.data_description()
        print("-" * 50)
        self.statistics()

    def apply_get_dummies(self, column_name):

        dummies = pd.get_dummies(
            self.df[column_name], prefix=column_name, drop_first=True)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[column_name], inplace=True)
        print(
            f"Applied pd.get_dummies to column '{column_name}' and dropped the first dummy variable.")


#   -------------------------------------------------------------------------------------- Data Visulization ------------------------------------------------------------------

    def plot_histograms_plots(self, x):
        sns.histplot(self.df[x], kde=True)
        plt.title(f'Histogram of {x}')
        plt.xlabel(x)
        plt.ylabel('Frequency')
        plt.show()

    def plot_kde_plots(self, x):
        sns.kdeplot(self.df[x], shade=True)
        plt.title(f'KDE Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Density')
        plt.show()

    def plot_ecdf_plots(self, x):
        sns.ecdfplot(data=self.df, x=x)
        plt.title(f'ECDF Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Cumulative Probability')
        plt.show()

    def plot_regplots(self, x, y):
        sns.regplot(data=self.df, x=x, y=y)
        plt.title(f'Regression Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_pair_plots(self):
        sns.pairplot(self.df[self.numerical_data])
        plt.title('Pair Plot of Numerical Columns')
        plt.show()

    def plot_scatter_plots(self, x, y):
        sns.scatterplot(data=self.df, x=x, y=y)
        plt.title(f'Scatter Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_line_plots(self, x, y):
        sns.lineplot(data=self.df, x=x, y=y)
        plt.title(f'Line Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_box_plots(self, x):
        sns.boxplot(data=self.df, y=x)
        plt.title(f'Box Plot of {x}')
        plt.ylabel(x)
        plt.show()

    def plot_count_plots(self, x):
        sns.countplot(data=self.df, x=x)
        plt.title(f'Count Plot of {x}')
        plt.xlabel(x)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    def plot_bar_plots(self, x, y):
        sns.barplot(data=self.df, x=x, y=y)
        plt.title(f'Bar Plot of {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=90)
        plt.show()

    def plot_point_plots(self, x, y):
        sns.pointplot(data=self.df, x=x, y=y)
        plt.title(f'Point Plot of {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=90)
        plt.show()

#   -------------------------------------------------------------------------------------- Machine Learning ------------------------------------------------------------------

    def evaluate_regression(self, y_column):
        warnings.filterwarnings(
            "ignore", category=FutureWarning)
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

            for model_name, accuracy in model_accuracies.items():
                print(f'{model_name} has RMSE: {accuracy}')
                print("-" * 50)

            best_model = min(model_accuracies,
                             key=lambda x: model_accuracies[x])
            print("-" * 50)
            print("-" * 50)
            print("The best model is:", best_model)
            print("-" * 50)
            print("-" * 50)

        except ValueError as ve:
            print(
                f"ValueError: {ve}. Make sure all columns '{self.df.columns}' is numerical for regression or preprocess the data.")
        except Exception as e:
            print(f"An error occurred: {e}.")

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

            for model_name, accuracy in model_accuracies.items():
                print(f'{model_name} has Mean RMSE: {accuracy}')
                print("-" * 50)
            best_model = min(model_accuracies,
                             key=lambda x: model_accuracies[x])
            print("-" * 50)
            print("-" * 50)
            print("The best model is:", best_model)
            print("-" * 50)
            print("-" * 50)

        except ValueError as ve:
            print(
                f"ValueError: {ve}. Make sure all columns '{self.df.columns}' is numerical for regression or preprocess the data.")
        except Exception as e:
            print(f"An error occurred: {e}.")

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

            for model_name, model in models.items():

                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                print("-" * 50)
                print(f"Model: {model_name}")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print("-" * 50)
        except:
            print(f'The {y_column} should be categorical or discrete in nature')

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
                print("-" * 50)
                print(f"Model: {model_name}")
                print(f"Accuracy: {accuracy_mean:.2f}")
                print(f"Precision: {precision_mean:.2f}")
                print(f"Recall: {recall_mean:.2f}")
                print(f"F1 Score: {f1_mean:.2f}")
                print("-" * 50)
        except:
            print(f'The {y_column} should be categorical or discrete in nature')

    def run_selected_function(self):

        function_mapping = {
            '1': self.data_description,
            '2': self.statistics,
            '3': self.percentage_of_outliers_in_columns,
            '4': self.handling_missing_values,
            '5': self.remove_outliers,
            'h': self.plot_histograms_plots,
            'k': self.plot_kde_plots,
            'e': self.plot_ecdf_plots,
            'r': self.plot_regplots,
            'a': self.plot_pair_plots,
            's': self.plot_scatter_plots,
            'l': self.plot_line_plots,
            'b': self.plot_box_plots,
            'c': self.plot_count_plots,
            'm': self.plot_bar_plots,
            'p': self.plot_point_plots,
            '6': self.evaluate_regression,
            '7': self.evaluate_regression_with_cv,
            '8': self.evaluate_classification_models,
            '9': self.evaluate_classification_models_with_cv,
            'd': self.apply_get_dummies,
            '0': exit}

        while True:
            print("Press the key corresponding to the function you want to run:")
            print("-" * 50)
            print("\n")
            print("1 - Data Description (Data Overview, Data Types Overview, Columns Names, Columns with Missing Values, Number of Unique Categories, Percentage of Missing Values)")
            print("2 - Statistics (Correlation, Coefficient of Variation, Descriptive Statistics, Skewness, Kurtosis)")
            print("3 - Percentage of Outliers in Columns")
            print("4 - Handling Missing Values remove or fill")
            print("5 - Remove Outliers")
            print("h - Plot Histograms")
            print("k - Plot KDE plots")
            print("e - Plot ECDF plots")
            print("r - Plot Regression plots *")
            print("a - Plot pair plots")
            print("s - Plot scatter plots *")
            print("l - Plot line plots *")
            print("b - Plot box plots")
            print("c - Plot count plots")
            print("m - Plot bar plots *")
            print("p - Plot point plots *")
            print("6 - Evaluate Regression")
            print("7 - Evaluate Regression with Cross-Validation")
            print("8 - Evaluate Classification Models")
            print("9 - Evaluate Classification Models with Cross-Validation")
            print("d - Apply pd.get_dummies to a Column")
            print("0 - Quit")
            print("*" * 70)
            try:
                key = keyboard.read_event(suppress=True).name
                if key == '0':
                    print("Quitting...")
                    break

                if key in ['1', '2', '3', '4', '5', 'a']:
                    function_mapping[key]()

                elif key in ['h', 'k', 'e', 'b']:
                    print(
                        f"Available numerical columns: {list(self.numerical_data)}")
                    x = input(
                        "Enter the numerical column name for selected plot: ")
                    if x in self.numerical_data:
                        function_mapping[key](x)
                    else:
                        print("+" * 50)
                        print("Invalid numerical column name.")
                    continue

                elif key in ['r', 's', 'l', ]:
                    print(
                        f"Available numerical columns: {list(self.numerical_data)}")
                    x_col = input(
                        "Enter the numerical column name for x-axis: ")
                    y_col = input(
                        "Enter the numerical column name for y-axis: ")
                    if (x_col in self.numerical_data) and (y_col in self.numerical_data):
                        function_mapping[key](x_col, y_col)
                    else:
                        print("+" * 50)
                        print("Invalid numerical column name")

                elif key in ['m', 'p']:
                    print(
                        f"Available categorical columns: {list(self.categorical_data)}")
                    print(
                        f"Available numerical columns: {list(self.numerical_data)}")
                    x_col = input(
                        "Enter the categorical column name for x-axis: ")
                    y_col = input(
                        "Enter the numerical column name for y-axis: ")
                    if (x_col in self.categorical_data) and (y_col in self.numerical_data):
                        function_mapping[key](x_col, y_col)
                    else:
                        print("+" * 50)
                        print(
                            "Invalid column names. x should be categorical and y should be numerical")

                elif key == 'c':
                    print(
                        f"Available categorical columns: {list(self.categorical_data)}")
                    x = input(
                        "Enter the categorical column name for selected plot: ")
                    if x in self.categorical_data:
                        function_mapping[key](x)
                    else:
                        print("+" * 50)
                        print("Invalid categorical column name")

                elif key in ['6', '7']:
                    if key == '6':
                        print(
                            f"Available numerical columns: {list(self.numerical_data)}")
                        y_col = input(
                            "Enter the numerical target column name: ")
                        if y_col in (self.numerical_data):
                            function_mapping[key](y_col)
                        else:
                            print("Invalid numerical column name")
                    else:
                        print(
                            f"Available numerical columns: {list(self.numerical_data)}")
                        y_col = input(
                            "Enter the numerical target column name: ")
                        num_folds = int(
                            input("Enter the number of cross-validation folds: "))
                        if (num_folds in range(1, 11)) and (y_col in (self.numerical_data)):
                            function_mapping[key](y_col, num_folds)
                        else:
                            print(
                                'Invalid input. Please enter an integer for the number of folds between (1-10) and check for Column Name')

                elif key in ['8', '9']:

                    if key == '8':
                        print(f'Available columns: {list(self.df.columns)}')
                        y_col = input("Enter the target column name: ")
                        if y_col in (self.df.columns):
                            function_mapping[key](y_col)
                        else:
                            print("Invalid column name")

                    else:
                        print(f'Available columns: {list(self.df.columns)}')
                        y_col = input("Enter the target column name: ")
                        num_folds = int(
                            input("Enter the number of cross-validation folds: "))
                        if (num_folds in range(1, 11)) and (y_col in (self.df.columns)):
                            function_mapping[key](y_col, num_folds)
                        else:
                            print(
                                'Invalid input. Please enter an integer for the number of folds between (1-10)')

                elif key == 'd':
                    print(
                        f"Available categorical columns: {list(self.categorical_data)}")
                    column_name = input(
                        "Enter the column name to apply pd.get_dummies: ")
                    if column_name in self.categorical_data:
                        function_mapping[key](column_name)
                    else:
                        print(
                            f"Column '{column_name}' not found categorical columns.")

                else:
                    print('=' * 50)
                    print("Invalid key. Press a valid key to run a function.")
                    print('=' * 50)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


# def main():
#     st.title("Automatid EDA and Machine Learning")

#     file_path = st.sidebar.file_uploader(
#         "Upload a file", type=["csv", "xls", "xlsx", "dta"])
#     if file_path is not None:
#         analyzer = Pycharet(file_path)

#         st.sidebar.subheader("Select a function:")
#         selected_function = st.sidebar.selectbox("", ["Data Description", "Statistics", "Percentage of Outliers", "Handling Missing Values", "Remove Outliers", "Plot Histograms", "Plot KDE plots", "Plot ECDF plots", "Plot Regression plots", "Plot Pair plots", "Plot Scatter plots",
#                                                  "Plot Line plots", "Plot Box plots", "Plot Count plots", "Plot Bar plots", "Plot Point plots", "Evaluate Regression", "Evaluate Regression with CV", "Evaluate Classification Models", "Evaluate Classification Models with CV", "Apply pd.get_dummies", "Quit"])

#         if selected_function == "Data Description":
#             analyzer.data_description()
#         elif selected_function == "Statistics":
#             analyzer.statistics()
#         elif selected_function == "Percentage of Outliers":
#             analyzer.percentage_of_outliers_in_columns()
#         # Add more function selections and calls here


# if __name__ == "__main__":
#     main()


def main():
    file = input('Add your File Path: ')
    file_path = os.path.join(file)
    print("\n")

    try:
        analyzer = Pycharet(file_path)
        analyzer.run_selected_function()
    except:
        print("File not found. Enter the correct path.")


if __name__ == "__main__":
    main()
