# Start with Docker

```bash
# to run locally with docker
./run_docker_locally.sh

```

# Install manually
You will require Python 3.10. `--workers 1` is required, until decent way to persist data is found.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --workers 1
```