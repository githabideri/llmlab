# LLMlab Benchmark Web UI

Web interface for running context ladder benchmarks on llama.cpp servers.

## Features

- 🤖 **Model Dashboard**: View current model, server status, and slot utilization
- 🚀 **Interactive Benchmarks**: Configure and run context ladder tests with live output streaming
- 📈 **Results Browser**: View and download previous benchmark results
- ⚙️ **Settings**: Display current server configuration
- 🎨 **Modern UI**: Styled to match llama.cpp's web interface (Tailwind CSS + dark theme)

## Quick Start

### 1. Install Dependencies

Create a virtual environment and install dependencies:

```bash
cd /var/lib/clawdbot/workspace/agents/labmaster/llmlab/web
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Configure (Optional)

Set environment variables to customize the setup:

```bash
export LLAMA_SERVER_URL="http://192.168.0.27:8080"  # Default
export PORT=8000                                      # Default
export HOST="0.0.0.0"                                # Default
```

### 3. Run the Server

**Recommended** - Use the start script:

```bash
./start.sh
```

Or activate the venv and run manually:

```bash
source venv/bin/activate
python app.py
```

Or with uvicorn directly (with auto-reload for development):

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the UI

Open your browser to:
- **Local**: http://localhost:8000
- **Network**: http://<your-ip>:8000

## Usage

### Dashboard
- View current model and server status
- Monitor slot utilization in real-time
- Check server connectivity

### Run Benchmark
1. Configure test points (comma-separated token counts)
2. Select output formats (JSON, CSV, Markdown)
3. Enable/disable resource monitoring
4. Click "Start Benchmark"
5. Watch live output as the test runs
6. Results are automatically saved to `scripts/results/`

### Results
- Browse previous benchmark runs
- View metadata (model, timestamp, test points)
- Download JSON results

### Settings
- View current server configuration
- Reference for ctx_size, parallel slots, GPU layers
- Note: Configuration changes require manual server restart

## Architecture

- **Backend**: FastAPI with async support
- **Frontend**: HTML + Tailwind CSS + htmx
- **Streaming**: Server-Sent Events (SSE) for live progress
- **Integration**: Calls the existing `run_context_ladder.py` script

## API Endpoints

- `GET /` - Serve the web UI
- `GET /api/models` - Get current model and server info
- `GET /api/settings` - Get server settings
- `GET /api/results` - List previous benchmark results
- `GET /api/results/{filename}` - Get specific result file
- `POST /api/benchmark/run` - Start benchmark with SSE streaming
- `GET /api/health` - Health check

## Development

### Run with auto-reload

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Add custom static files

Place CSS/JS files in `static/` and they'll be served at `/static/*`

### Customize templates

Edit `templates/index.html` - uses Jinja2 templating

## Troubleshooting

### Server connection failed
- Check that llama.cpp server is running at the configured URL
- Verify firewall rules allow connection
- Check `LLAMA_SERVER_URL` environment variable

### Benchmark won't start
- Ensure `run_context_ladder.py` is executable
- Check that Python3 is available
- Verify `.env` file exists in `scripts/` directory

### Live output not streaming
- Some reverse proxies buffer SSE - configure `X-Accel-Buffering: no`
- Check browser console for connection errors

## Production Deployment

For production use, run behind a reverse proxy:

### nginx example

```nginx
location /bench/ {
    proxy_pass http://localhost:8000/;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
}
```

### systemd service

Create `/etc/systemd/system/llmlab-bench.service`:

```ini
[Unit]
Description=LLMlab Benchmark Web UI
After=network.target

[Service]
Type=simple
User=clawdbot
WorkingDirectory=/var/lib/clawdbot/workspace/agents/labmaster/llmlab/web
Environment="LLAMA_SERVER_URL=http://192.168.0.27:8080"
Environment="PORT=8000"
ExecStart=/var/lib/clawdbot/workspace/agents/labmaster/llmlab/web/venv/bin/python app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llmlab-bench
sudo systemctl start llmlab-bench
```

## License

MIT License - see parent directory LICENSE file
