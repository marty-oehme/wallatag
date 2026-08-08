# Dokku hosts the app AND its Prefect worker; the Prefect server schedules runs.
# The release phase registers the deployment (and ensures the work pool) on
# every deploy, before the worker starts.
release: python deploy/release.py
worker: prefect worker start -p wallatag-pool --type process
