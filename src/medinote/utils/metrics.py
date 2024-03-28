from pandas import DataFrame
from sklearn.metrics import classification_report

from medinote import initialize
from medinote.cached import read_dataframe

config, logger = initialize()

def measure_metrics(df: DataFrame = None,
                    y_true_column: str = 'y_true',
                    y_pred_column: str = 'y_pred',
                    label_names_column: str = 'label_names'):
    if df is None:
        input_path = config.metrics['input_path']
        df = read_dataframe(input_path)

    y_true = df[y_true_column]
    y_pred = df[y_pred_column]
    target_names = df[label_names_column]
    report = classification_report(y_true, y_pred, target_names=target_names)
    logger.info("Classification Report:\n" + report)
    
if __name__ == '__main__':
    measure_metrics()