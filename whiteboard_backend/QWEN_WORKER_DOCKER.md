# Qwen Worker Docker Container

This is a second container for the Qwen/vLLM worker. It is separate from the
Postgres container.

## Start the server

From `whiteboard_backend`:

```powershell
docker compose -f docker-compose.qwen.yml up --build -d
```

The server listens on:

```text
http://127.0.0.1:8009
```

## Watch logs

```powershell
docker logs -f drawnout_qwen_worker
```

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8009/health
```

## Load the text model

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8009/load/text `
  -ContentType "application/json" `
  -Body '{"model":"cyankiwi/Qwen3.5-4B-AWQ-4bit","gpu_memory_utilization":0.90,"max_model_len":8192,"warmup":true}'
```

## Check status

```powershell
Invoke-RestMethod http://127.0.0.1:8009/status
```

## Stop it

```powershell
docker compose -f docker-compose.qwen.yml down
```

The Hugging Face model cache is stored in the Docker volume `qwen_hf_cache`, so
models do not redownload every time the container restarts.
