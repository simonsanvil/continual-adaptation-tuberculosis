from typing import List, Dict, Any
import pandas as pd

__all__ = ["db_to_pandas", "pandas_to_db"]


def to_df(db_objects:List[Any], dict_factory=dict, nested=False) -> pd.DataFrame:
    """
    Convert a list of database objects to a pandas DataFrame.
    """
    return pd.DataFrame(map(lambda x: dataclass_to_dict(x, nested=nested), db_objects))

def pandas_to_db(df:pd.DataFrame, db_model:Any) -> List[Any]:
    """
    Convert a pandas DataFrame to a list of database objects.
    """
    return [db_model(**row) for _, row in df.iterrows()]

def dataclass_to_dict(obj, dict_factory=dict, nested=False):
    """
    Does the same as dataclass.asdict but won't be applied
    recursively to iterables
    """
    from dataclasses import is_dataclass, fields

    if is_dataclass(obj):
        result = []
        for f in fields(obj):
            if nested:
                value = dataclass_to_dict(getattr(obj, f.name), dict_factory)
            else:
                value = getattr(obj, f.name)
            result.append((f.name, value))
        return dict_factory(result)
    return obj