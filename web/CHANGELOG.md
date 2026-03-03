# Changelog

## [0.1.0] - 2026-03-03

### Added
- Initial release of LLMlab Benchmark Web UI
- FastAPI backend with SSE streaming for live benchmark progress
- Modern dark-themed UI matching llama.cpp web interface styling
- Dashboard view with model info, server status, and slot monitoring
- Interactive benchmark configuration and execution
- Results browser for viewing previous test runs
- Server settings display
- Health check monitoring
- htmx-based frontend for lightweight interactivity
- Tailwind CSS for responsive, modern styling
- Virtual environment setup with all dependencies
- Startup script (`start.sh`) for easy launching
- Comprehensive README with setup and deployment instructions

### Features
- Auto-query available models from llama.cpp server
- Configure test points and output formats
- Real-time streaming of benchmark output
- Save results in JSON, CSV, and Markdown formats
- View historical benchmark results
- Monitor server health and connectivity
- Display current server configuration

### Technical Details
- Python 3.12+ compatible
- FastAPI + uvicorn for async backend
- httpx for async HTTP requests
- Jinja2 templating
- Server-Sent Events (SSE) for progress streaming
- Integrates with existing `run_context_ladder.py` script
