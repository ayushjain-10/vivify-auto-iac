# Vibe DevOps Backend API

FastAPI backend for GCP resource discovery and task management.

## Features

- 🔍 **GCP Resource Discovery** - Discover and catalog GCP infrastructure
- ✅ **Credential Validation** - Validate service account credentials
- 📊 **Resource Enrichment** - Add metrics, costs, and health status
- 🔗 **Relationship Detection** - Map connections between resources
- 🎯 **Application Stacks** - Group resources by application

## Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

## Quick Start

1. **Activate virtual environment** (if not already activated):
   ```bash
   source ../venv/bin/activate  # macOS/Linux
   # or
   ..\venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python main.py
   ```

   Or with uvicorn directly:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## API Endpoints

### Health Check
```
GET /health
```

### GCP Discovery

**Validate Credentials**
```
POST /api/gcp/validate-credentials
Body: {
  "credentials": { ...service account JSON... }
}
```

**Discover Resources**
```
POST /api/gcp/discover
Body: {
  "credentials": { ...service account JSON... },
  "project": "my-project-id",  // optional
  "regions": ["us-central1"]   // optional
}
```

**Get Cached Architecture**
```
GET /api/gcp/architecture/{project}
```

**Clear Cache**
```
DELETE /api/gcp/architecture/{project}
```

## Project Structure

```
vivify-backend/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── api/
│   ├── routes/
│   │   └── gcp.py         # GCP endpoints
│   └── models/
│       └── gcp.py         # Pydantic models
├── services/
│   └── gcp_discovery.py   # Discovery logic
└── utils/
    └── auth.py            # Authentication utilities
```

## Environment Variables

Create a `.env` file:

```bash
PORT=8000
HOST=0.0.0.0
DEBUG=True
FRONTEND_URL=http://localhost:3000
```

## Development

### Run with auto-reload
```bash
uvicorn main:app --reload --port 8000
```

### View API documentation
Open http://localhost:8000/docs in your browser

### Test endpoints
Use the interactive docs or curl:

```bash
# Health check
curl http://localhost:8000/health

# Validate credentials
curl -X POST http://localhost:8000/api/gcp/validate-credentials \
  -H "Content-Type: application/json" \
  -d '{"credentials": {...}}'
```

## GCP Permissions Required

The service account needs these roles:
- **Viewer** (roles/viewer) - Read access to all resources
- Or specific roles:
  - Compute Viewer
  - Storage Object Viewer
  - Kubernetes Engine Viewer
  - Monitoring Viewer

## Troubleshooting

### Import errors
```bash
# Make sure you're in the virtual environment
source ../venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Port already in use
```bash
# Change port in .env or use different port
uvicorn main:app --reload --port 8001
```

### GCP authentication errors
- Verify service account has required permissions
- Check credentials JSON is valid
- Ensure project ID is correct

## Next Steps

- [ ] Add task management endpoints
- [ ] Implement WebSocket for real-time updates
- [ ] Add caching with Redis
- [ ] Implement database for persistence
- [ ] Add authentication/authorization
- [ ] Add rate limiting
- [ ] Add logging and monitoring

## License

MIT
