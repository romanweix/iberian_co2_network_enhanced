import pandas as pd
from pathlib import Path

def save_dict_series(path, data, index_name="index", value_name="value"):
    """Save a {key: value} dict to a 2-column CSV."""
    df = pd.DataFrame(list(data.items()), columns=[index_name, value_name])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
