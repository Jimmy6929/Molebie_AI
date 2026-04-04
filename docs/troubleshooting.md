# Troubleshooting

Run `molebie-ai doctor` for a full diagnostic — it checks dependencies, config files, and service health with suggested fixes.

| Problem | Solution |
|---------|----------|
| Something seems wrong | `molebie-ai doctor` — checks everything and suggests fixes |
| Missing `.env.local` or config | `molebie-ai doctor --fix` or just `molebie-ai run` (auto-creates both) |
| Address already in use | `molebie-ai run` auto-kills stale processes; or `lsof -i :<port>` then `kill` |
| Gateway crashes | Check `data/logs/gateway.log` — error details logged there |
| Auth 401 errors | `molebie-ai config get JWT_SECRET --show-secrets` to inspect |
| Voice transcription fails | `brew install ffmpeg` |
| OMP error on macOS | `make dev-gateway` sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically |
| Config looks wrong | `molebie-ai config env` to list all vars, `molebie-ai config set KEY=VALUE` to fix |

## System Requirements

- **OS**: macOS, Linux, or Windows (WSL2)
- **RAM**: 8GB minimum, 16GB+ recommended for local inference
- **GPU**: Recommended for local LLM inference (Apple Silicon, NVIDIA)
- **Disk**: ~2-10GB per model depending on quantization
- **Docker**: Optional, only needed for SearXNG web search and Kokoro TTS
