import numpy as np
import pandas as pd


def sigmoid_normalization(data, mean, std):
    """Apply sigmoid normalization to a data series."""
    return 1 / (1 + np.exp(-(data - mean) / std))


def convert_to_numeric(df, columns):
    """
    Convert specified columns to numeric, handling errors.

    Args:
        df (pd.DataFrame): DataFrame to modify
        columns (list): Columns to convert to numeric
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


class ParameterProcessor:
    def __init__(self, data, name, groups=None, type="grouped", location="nodes"):
        """
        Processor for grouped and ungrouped parameters.

        Args:
            data (gpd.GeoDataFrame): DataFrame containing the data.
            name (str): Name of the parameter (e.g., "rainfall", "visibility", "betweenness_centrality").
            groups (dict, optional): Grouping logic for "grouped" parameters (e.g., seasons and their months). Default is None.
            type (str): Type of parameter - "grouped" or "ungrouped". Default is "grouped".
            location (str): Location of the data (default: "nodes").
        """
        self.data = data
        self.name = name
        self.groups = groups
        self.type = type
        self.location = location
        self.field_name = f"avg_{name}" if type == "grouped" else name  # Handle grouped vs ungrouped field names.

    def process(self):
        """
        Process the parameter: grouping, averaging, and normalization (for grouped or ungrouped data).
        """
        if self.type == "grouped":
            self.process_grouped()
        elif self.type == "ungrouped":
            self.process_ungrouped()
        return self.data

    def process_grouped(self):
        """
        Process grouped parameters (e.g., rainfall, visibility) using grouping logic.
        """
        # Convert fields to numeric
        parameter_columns = [
            f"{self.field_name}_{month}" for months in self.groups.values() for month in months
        ]
        convert_to_numeric(self.data, parameter_columns)

        # Process each group
        for group_name, group_items in self.groups.items():
            # Calculate the mean for the group (e.g., for a season)
            group_data = self.data[[f"{self.field_name}_{item}" for item in group_items]].mean(axis=1)

            # Compute mean and standard deviation
            group_mean = group_data.mean()
            group_std = group_data.std()

            # Create normalized column
            self.data[f"normalized_{self.name}_{group_name}"] = sigmoid_normalization(group_data, group_mean, group_std)

            # Store seasonal statistics
            self.data[f"{self.name}_mean_{group_name}"] = group_mean
            self.data[f"{self.name}_std_{group_name}"] = group_std

        # Drop intermediate columns
        # self.cleanup_intermediate_columns(parameter_columns)

    def process_ungrouped(self):
        """
        Process ungrouped parameters (e.g., betweenness_centrality).
        """
        convert_to_numeric(self.data, [self.field_name])

        # Ensure the column exists
        if self.field_name not in self.data.columns:
            raise ValueError(f"Column {self.field_name} not found in the data.")

        # Compute mean and standard deviation
        column_data = self.data[self.field_name]
        column_mean = column_data.mean()
        column_std = column_data.std()

        # Create normalized column
        self.data[f"normalized_{self.name}"] = sigmoid_normalization(column_data, column_mean, column_std)

    def cleanup_intermediate_columns(self, columns):
        """Drop the intermediate columns after processing."""
        self.data.drop(columns=columns, inplace=True)


