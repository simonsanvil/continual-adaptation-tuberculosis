import os, pickle, joblib

from 

def load_model_from_source(model_source:str, model_uri:str, **kwargs):
    if model_source == "local":
        if "serializer" in kwargs:
            serializer = kwargs["serializer"]
            if serializer == "pickle":
                model = pickle.load(model_uri)
            elif serializer == "joblib":
                model = joblib.load(model_uri)
            else:
                raise Exception("Unknown serializer.")
        else:
            model = pickle.load(model_uri)
    elif model_source == "s3":
        model = load_model_from_s3(model_uri)
    elif model_source == "mlflow":
        model = load_model_from_mlflow(model_uri)
    elif model_source == "keras":
        from keras.models import load_model
        model = load_model(model_uri)
    elif model_source == "cloud":
        import cloudpathlib
        with cloudpathlib.CloudPath(model_uri).open("rb") as f:
            model = pickle.load(f)
    else:
        raise Exception("Unknown model source.")
    return model

def deserialize_model(f:Io, serializer:str):


def load_model_from_s3(model_uri:str):
    pass

def load_model_from_mlflow(model_uri:str):
    import mlflow
    model = mlflow.pyfunc.load_model(model_uri)
    return model
