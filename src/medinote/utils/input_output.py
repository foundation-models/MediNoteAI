


from pandas import DataFrame, read_csv, read_parquet
from medinote import initialize
import os
from pandas.errors import EmptyDataError


config, logger = initialize()

def read_input_dataframe(input_path: str, do_create_dataframe: bool = False):
    """_summary_

    Prepare data for inference.
    """
    if input_path:
        if do_create_dataframe and not os.path.exists(input_path):
            return DataFrame()
        try:
            if input_path.endswith('.parquet'):
                df = read_parquet(input_path)
            elif input_path.endswith('.csv'):
                df = read_csv(input_path)
            elif input_path.endswith('.txt'):
                df = read_csv(input_path, sep='\t', header=None, names=['text'])
            else:
                raise ValueError(
                    f"Unsupported file format for {input_path}. Only .parquet, .csv, and .txt are supported.")
            df.drop_duplicates(inplace=True)
            logger.info(f"Read {len(df)} rows from {input_path}")
            return df
        except EmptyDataError as e:
            if do_create_dataframe:
                return DataFrame()
            else:
                raise e
        except Exception as e:
            raise e
    else:
        raise ValueError(f"input_path is not provided")


def write_output_dataframe(df, output_path: str):
    """_summary_

    Write data to a file.
    """
    if output_path:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        if output_path.endswith('.parquet'):
            df.to_parquet(output_path)
        elif output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file format for {output_path}. Only .parquet and .csv are supported.")
        logger.info(f"Written {len(df)} rows to {output_path}")
    else:
        raise ValueError(f"output_path is not provided")