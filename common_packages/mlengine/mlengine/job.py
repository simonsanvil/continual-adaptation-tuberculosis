from abc import ABC, abstractmethod

class Job(ABC):
    
    def __init__(self, job_id):
        self.job_id = job_id
        self._job = None

    def run(self):
        job = self._get_job()
        job()

    def _get_job(self):
        if self._job is None:
            self._job = self._get_job_from_db()
        
        return self._job