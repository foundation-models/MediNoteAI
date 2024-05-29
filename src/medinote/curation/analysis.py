
import os
from pandas import DataFrame
import yaml
from medinote import read_dataframe


try:
    with open(
        os.environ.get("CONFIG_YAML"), "r"
    ) as file:
        conf = yaml.safe_load(file)
        analysis_config = conf.get("analysis")
        if not analysis_config:
            raise Exception("Analysis config not found.")


    def analysis(df: DataFrame = None, config: dict = None):
        """
        Perform analysis on the data.

        Args:
            df (DataFrame): The DataFrame containing the data.
            config (dict): The configuration dictionary.

        Returns:
            DataFrame: The DataFrame containing the analysis results.
        """
        config = config = analysis_config
        # Perform analysis
        if df is None:
            # Perform analysis on the data
            input_path = config.get("input_path")
            if input_path:
                df = read_dataframe(input_path)
            else:
                raise ValueError("Input path not provided.")
            
        report_columns = config.get("report_columns") or []
        report = {}
        for report_column in report_columns:
            if report_column not in df.columns:
                raise ValueError(f"Column '{report_column}' not found in the DataFrame.")
            report[report_column] = df[report_column].nunique()
        
        output_path = config.get("output_path")
        return DataFrame(report, index=[0]).to_csv(output_path, index=False)
except Exception as e:
    raise e    

if __name__ == "__main__":
    analysis()