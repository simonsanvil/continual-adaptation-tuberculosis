from typing import Dict
import logging

from ..job import Job

class InferenceJob(Job):

    def __init__(
        self,
        job_id: str,
        model_loading: callable,
        predict_function: callable,
        config_path: str,
    ):
        super().__init__(job_id)
        self._model_loading = model_loading
        self._predict_function = predict_function
        self._config_path = config_path
        self._job = self.run

    @property
    def config_path(self):
        return self._config_path
    
    def _load_config(self):
        import yaml
        from ml_collections import config_dict
        config = config_dict.ConfigDict(yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader))
        return config
    
    def run(self):
        import uvicorn

        config = self._load_config()
        app = self.make_predict_endpoint(config)
        uvicorn.run(app, host=config.inference_host, port=config.inference_port)
        
    def make_predict_endpoint(self, config:dict=None) -> "Starlette":
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        import dotenv

        if config is None:
            config = self._load_config()
        
        dotenv.load_dotenv(config.environment.dotenv_path)
        model = self._model_loading(config)
        logging.info(f"Successfully loaded model from {self.config_path}")

        async def predict_wrapper(request) -> Dict:
            data = await request.json()
            result = self._predict_function(model, data)
            return JSONResponse(result)
        
        async def check_health(request):
            return JSONResponse({"status": "ok"})
        
        app = Starlette(debug=True, routes=[
            Route("/predict", predict_wrapper, methods=["POST"]),
            Route("/health", check_health, methods=["GET"])
        ])

        return app

