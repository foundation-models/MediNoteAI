
from medinote import initialize, merge_parquet_files
import sys
import os
import glob


config, logger = initialize()

def remove_files_with_pattern(pattern: str):
    file_list = glob.glob(pattern)
    for file in file_list:
        os.remove(file)

def merge_all_sqlcoder_files(pattern: str = None,
                             output_path: str = None,
                             obj_name: str = None
                             ):
    pattern = pattern or config.sqlcoder.get('output_prefix') + '*'

    output_path = output_path or config.sqlcoder.get('merge_output_path')

    if obj_name:
        pattern = pattern.replace('SRC', obj_name)
        output_path = output_path.replace('SRC', obj_name)

    logger.info(f"Merging all SQLCoder files to {output_path}")
    df = merge_parquet_files(pattern, identifier_index=1, identifier=obj_name)
    to_remove_columns = ['totalRecords',
                         '__index_level_0__', 'text:1', '__index_level_0__:1']
    for col in to_remove_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)

    input_column = config.sqlcoder.get('input_column')
    df = df[~df[input_column].str.contains(
        "do not know", case=False, na=False)]

    df.to_parquet(output_path)
    remove_files_with_pattern(pattern)


def merge_all_screened_files(pattern: str = None,
                             output_path: str = None,
                             obj_name: str = None,

                             ):
    pattern = pattern or config.screening.get('output_prefix') + '*'
    output_path = output_path or config.screening.get('merge_output_path')
    df = merge_parquet_files(pattern, identifier=obj_name)
    logger.info(f"Merging all Screening files to {output_path}")
    df.to_parquet(output_path)
    remove_files_with_pattern(pattern)


def merge_all_pdf_reader_files(pattern: str = None,
                             output_path: str = None,
                             ):
    pattern = pattern or config.pdf_reader.get('merge_pattern') or config.pdf_reader.get('output_prefix') + '*'
    output_path = output_path or config.pdf_reader.get('merge_output_path')
    df = merge_parquet_files(pattern)
    logger.info(f"Merging all Screening files to {output_path}")
    df.to_parquet(output_path)
    remove_files_with_pattern(pattern)

def merge_all_embedding_files(pattern: str = None, 
                             output_path: str = None):
    pattern = pattern or config.embedding.get('output_prefix') + '*'
    output_path = output_path or config.embedding.get('output_path')
    df = merge_parquet_files(pattern)
    df.to_parquet(output_path)
    remove_files_with_pattern(pattern)
    
def custom():
    pattern = '/mnt/datasets/archive/parquet/sqlcoder/sqlcoder_assetfi*.parquet'
    df = merge_parquet_files(pattern)
    df.to_parquet('/mnt/aidrive/datasets/sql_gen/tmp.parquet')
    remove_files_with_pattern(pattern)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == 'screening':
            merge_all_screened_files(obj_name='asset')
        elif step == 'embedding':
            merge_all_embedding_files()
        elif step == 'sqlcoder':
            merge_all_sqlcoder_files(obj_name='asset')
        elif step == 'pdf_reader':
            merge_all_pdf_reader_files()
        else:
            print("Invalid step argument. Please choose 'screening' or 'sqlcoder'.")
    else:
        merge_all_screened_files()
        print("Please provide a step argument. Choose 'screening' or 'sqlcoder'.")
