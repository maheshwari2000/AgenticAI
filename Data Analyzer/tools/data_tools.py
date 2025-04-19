from langchain.tools import Tool
import pandas as pd
import json
from typing import Optional

csv_data: Optional[pd.DataFrame] = None

# Tool 1: Read CSV
def read_csv(file_path: str) -> str:
    """Loads a CSV file and returns basic details."""
    global csv_data
    try:
        csv_data = pd.read_csv(file_path)
        return json.dumps({"message": "CSV loaded successfully", "rows": csv_data.shape[0], "columns": csv_data.shape[1]})
    except Exception as e:
        return json.dumps({"error": str(e)})

read_csv_tool = Tool(name="read_csv", func=read_csv, description="Loads a CSV file and returns number of rows and columns.")

# Tool 2: Check Data Types
def check_data_types(_input: Optional[str] = None) -> str:
    """Checks data types of all columns."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    
    types = csv_data.dtypes.apply(lambda x: str(x)).to_dict()
    return json.dumps({"data_types": types})

check_data_types_tool = Tool(name="check_data_types", func=check_data_types, description="Displays the data types of all columns in the dataset.")

# Tool 3: Check for Duplicates
def check_duplicates(_input: Optional[str] = None) -> str:
    """Finds duplicate rows in the dataset."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})

    try:
        df = pd.DataFrame(csv_data)
        duplicates = df[df.duplicated(keep=False)]

        return {
            "duplicate_count": int(df.duplicated().sum()),
            "duplicate_rows": duplicates.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}

check_duplicates_tool = Tool(name="check_duplicates", func=check_duplicates, description="Checks for duplicate rows in the dataset.")

# Tool 4: Clean Data
def clean_data(strategy: str = "mean") -> str:
    """Cleans dataset by removing duplicates and handling missing values based on a strategy (mean, median, mode)."""
    global csv_data
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    
    csv_data = csv_data.drop_duplicates()
    
    if strategy == "mean":
        csv_data = csv_data.fillna(csv_data.mean(numeric_only=True))
    elif strategy == "median":
        csv_data = csv_data.fillna(csv_data.median(numeric_only=True))
    elif strategy == "mode":
        csv_data = csv_data.fillna(csv_data.mode().iloc[0])
    
    return json.dumps({"message": "Data cleaned", "rows_after_cleaning": csv_data.shape[0]})

clean_data_tool = Tool(name="clean_data", func=clean_data, description="Cleans the dataset by removing duplicates and filling missing values with a chosen strategy (mean, median, mode).")

# Tool 5: Missing Data Analysis
def missing_data(_input: Optional[str] = None) -> str:
    """Calculates the percentage of missing values in each column."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    
    missing_percent = csv_data.isnull().sum() / len(csv_data) * 100
    return json.dumps({"missing_data_percentage": missing_percent.to_dict()})

missing_data_tool = Tool(name="missing_data", func=missing_data, description="Returns the percentage of missing data in each column.")

# Tool 6: Summary Statistics
def stat_data(_input: Optional[str] = None) -> str:
    """Returns statistics (mean, median, min, max) for numerical columns."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    
    summary = csv_data.describe().T.to_dict()
    return json.dumps({"statistics": summary})

stat_data_tool = Tool(name="stat_data", func=stat_data, description="Provides statistical summary of numerical columns.")

# Tool 7: Detect Outliers
def detect_outliers(column_name: str) -> str:
    """Detects outliers in a numerical column using the IQR method."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    
    if column_name not in csv_data.columns or not pd.api.types.is_numeric_dtype(csv_data[column_name]):
        return json.dumps({"error": f"Column '{column_name}' is not numeric or not found."})

    Q1 = csv_data[column_name].quantile(0.25)
    Q3 = csv_data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers = csv_data[(csv_data[column_name] < (Q1 - 1.5 * IQR)) | 
                        (csv_data[column_name] > (Q3 + 1.5 * IQR))]
    
    return json.dumps({"column": column_name, "outlier_count": len(outliers)})

detect_outliers_tool = Tool(name="detect_outliers", func=detect_outliers, description="Detects outliers in a given numerical column.")

# Tool 8: Correlation Matrix
def correlation_matrix(_input: Optional[str] = None) -> str:
    """Computes and returns the correlation matrix."""
    if csv_data is None:
        return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})
    numeric_columns = []
    for col in csv_data.columns:
        if pd.api.types.is_numeric_dtype(csv_data[col]):
            numeric_columns.append(col)
    correlation_matrix = csv_data[numeric_columns].corr().to_dict()
    return json.dumps({"correlation_matrix": correlation_matrix})

correlation_matrix_tool = Tool(name="correlation_matrix", func=correlation_matrix, description="Computes and displays the correlation matrix of numerical columns.")

# Tool 9: Value Counts for Categorical Columns
# def value_counts(column_name: str) -> str:
#     """Returns value counts for a categorical column."""
#     if csv_data is None:
#         return json.dumps({"error": "No CSV file is loaded yet. Load a CSV first."})

#     if column_name not in csv_data.columns or pd.api.types.is_numeric_dtype(csv_data[column_name]):
#         return json.dumps({"error": f"Column '{column_name}' is not categorical or not found."})

#     counts = csv_data[column_name].value_counts().to_dict()
#     return json.dumps({"column": column_name, "value_counts": counts})

# value_counts_tool = Tool(name="value_counts", func=value_counts, description="Returns value counts for a given categorical column.")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import io

# def visualize_data(column_name: str) -> str:
#     """Generates a histogram for a selected numerical column."""
#     if column_name not in csv_data.columns:
#         return f"Error: Column '{column_name}' not found in the CSV file."

#     # Create the plot
#     plt.figure(figsize=(8, 5))
#     sns.histplot(csv_data[column_name].dropna(), bins=20, kde=True)
#     plt.title(f"Distribution of {column_name}")
#     plt.xlabel(column_name)
#     plt.ylabel("Frequency")
    
#     # Save the plot as an image and return the path
#     img_path = "histogram.png"
#     plt.savefig(img_path)
#     plt.close()
#     return img_path  # Returns image path for display

# visualize_data_tool = Tool(name="visualize_data", func=visualize_data, description="Generates a histogram for a given numerical column.")

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# def train_model(target_column: str) -> str:
#     """Trains a simple regression model on the dataset."""
#     if target_column not in csv_data.columns or not pd.api.types.is_numeric_dtype(csv_data[target_column]):
#         return f"Error: Column '{target_column}' is not numeric or not found."

#     X = csv_data.drop(columns=[target_column]).select_dtypes(include=['number'])
#     y = csv_data[target_column]
    
#     if X.empty:
#         return "Error: No numerical columns found for training."

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     score = model.score(X_test, y_test)
#     return f"✅ Model trained successfully with R² score: **{score:.2f}**."

# train_model_tool = Tool(name="train_model", func=train_model, description="Trains a simple regression model using a selected target column.")