from pandas import DataFrame, Series, read_parquet
from medinote import initialize
from medinote import write_dataframe


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def generate_prompt(
    row: Series,
    template: str = None,
    samples_column: str = None,
    id_column: str = None,
    content_column: str = None,
    sql_column: str = None,
    sample_df: DataFrame = None,
    config: dict = None,
) -> str:

    template = template or config.get("finetune")["prompt_template"]
    samples_column = samples_column or config.get("finetune")["samples_column"]
    id_column = id_column or config.get("finetune")["id_column"]
    content_column = content_column or config.get("finetune")["content_column"]
    sql_column = sql_column or config.get("finetune")["sql_column"]
    if sample_df is None:
        sample_df = read_parquet(config.get("finetune")["input_path"])

    if samples_column and samples_column in row:
        samples = row[samples_column]
        row_dict = row.to_dict()
        if sample_df is not None:
            if content_column in sample_df.columns:
                subset_df = sample_df[sample_df[id_column].isin(samples)]
                row_dict["samples"] = [
                    "input: "
                    + str(row[content_column])
                    + "\t output:"
                    + str(row[sql_column])
                    for _, row in subset_df.iterrows()
                ]
                row_dict["samples"] = [
                    sample
                    for sample in row_dict["samples"]
                    if row[content_column] not in sample
                ]
                row_dict["samples"] = "\n".join(row_dict["samples"])
        else:
            logger.error(
                f"Invalid value for sample_df or id_column: {sample_df}, {id_column}"
            )
        template = template or config.get("finetune")["prompt_template"]
        logger.debug(f"Using template: {template}")
        prompt = template.format(**row_dict)
        return prompt
    else:
        logger.error(f"Column {samples_column} does not exist in row")
    return ""


def parallel_generate_prompt(
    df: DataFrame = None,
    template: str = None,
    samples_column: str = None,
    id_column: str = None,
    content_column: str = None,
    prompt_column: str = None,
    persist: bool = True,
    config: dict = None,
) -> DataFrame:

    df = df or read_parquet(config.get("finetune")["input_path"])
    # df = df[:4]
    prompt_column = prompt_column or config.get("finetune")["prompt_column"]
    output_path = config.get("finetune")["output_path"]

    logger.debug(f"Generating prompt for {df.shape[0]} rows")
    df[prompt_column] = df.apply(
        generate_prompt,
        axis=1,
        template=template,
        samples_column=samples_column,
        id_column=id_column,
        content_column=content_column,
        sample_df=df,
    )
    if persist and output_path:
        logger.debug(f"Saving to {output_path}")
        write_dataframe(df=df, output_path=output_path)

    return df


if __name__ == "__main__":
    parallel_generate_prompt()
