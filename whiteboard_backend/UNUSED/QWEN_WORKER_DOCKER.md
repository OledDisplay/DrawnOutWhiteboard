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

## Load the InternVL vision model

Use the report tester defaults when possible; they avoid slow CPU offload and
skip obsolete InternVL processor kwargs.

```powershell
python whiteboard_backend\test_cluster_label_report.py --processed-id processed_1 --depicts "eukaryotic cell diagram" --max-clusters 10
```

If you load vision manually, do not send `mm_processor_kwargs.max_dynamic_patch`
for InternVL through vLLM:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8009/load/vision `
  -ContentType "application/json" `
  -Body '{"model":"cyankiwi/InternVL3_5-8B-AWQ-4bit","processor":"OpenGVLab/InternVL3_5-8B-HF","gpu_memory_utilization":0.84,"max_model_len":1024,"max_num_batched_tokens":1024,"max_num_seqs":1,"cpu_offload_gb":0,"limit_mm_per_prompt":{"image":1},"disable_log_stats":true,"warmup":false}'
```

## Stop it

```powershell
docker compose -f docker-compose.qwen.yml down
```

The Hugging Face model cache is stored in the Docker volume `qwen_hf_cache`, so
models do not redownload every time the container restarts.

## Troubleshooting

### `POST /load/vision` returns 500; logs show `ValueError: Free memory on device ... is less than desired GPU memory utilization`

This is **not** “model not found.” vLLM checks that **free** VRAM is at least
`gpu_memory_utilization * total_VRAM`. On a **12GB** laptop GPU, asking for
**0.93** reserves ~**11.15 GiB**; if only **~10.8 GiB** is free (common right
after switching models or with driver overhead), startup **fails**.

- From `test_cluster_label_report.py`, pass a lower value, e.g.
  `--vision-worker-gpu-memory 0.82` or `0.80`.
- Close other GPU processes; then retry.

### “I have the model on disk but Docker still downloads”

The compose file mounts a **named volume** at `HF_HOME=/models/huggingface`, not
your host’s default `~/.cache/huggingface` unless you add a bind mount. Either
let the first run populate `qwen_hf_cache`, or mount your local cache into the
container.
