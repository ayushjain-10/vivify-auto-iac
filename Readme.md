# Vibe DevOps

> A modern cloud DevOps platform combining AI-powered task management with real-time GCP infrastructure visualization.

## What is Vibe DevOps?

Vibe DevOps is an intelligent platform that helps DevOps teams manage their cloud infrastructure and deployment tasks through an intuitive interface powered by AI. It combines three core capabilities:

1. **AI Chat Assistant** - A conversational agent powered by Google Gemini that helps you manage tasks, query infrastructure, and get DevOps insights
2. **Kanban Task Board** - Real-time task management with drag-and-drop functionality and WebSocket synchronization
3. **GCP Architecture Dashboard** - Visual representation of your Google Cloud Platform resources with live metrics, costs, and health monitoring

## Key Features

- 🤖 **Conversational AI Agent** - Natural language interface for task and infrastructure management
- 📋 **Real-time Kanban Board** - Live task updates using WebSocket with JSON Patch protocol
- ☁️ **Cloud Resource Discovery** - Automatic GCP infrastructure scanning and visualization
- 📊 **Live Metrics & Monitoring** - Real-time resource health, performance metrics, and cost tracking
- 🔄 **WebSocket Streaming** - Instant updates across all connected clients
- 🎨 **Modern Dark UI** - Clean, responsive interface with GCP brand colors
- 🔍 **Web Search Integration** - Agent can search the web for DevOps information
- 🛠️ **Tool-based Architecture** - Extensible agent with pluggable tools

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────────┬────────────────────────────────────────┐  │
│  │ Chat Panel   │  Kanban Board / GCP Dashboard          │  │
│  │ (Gemini AI)  │  (Real-time WebSocket Updates)         │  │
│  └──────────────┴────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI + Python)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangChain Agent (Gemini 2.0 Flash)                  │   │
│  │  ├─ Task Management Tool                             │   │
│  │  ├─ Canvas Query Tool (GCP Resources)                │   │
│  │  └─ Web Search Tool (Tavily)                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GCP Discovery Service                               │   │
│  │  ├─ Resource Discovery (Compute, Storage, GKE)       │   │
│  │  ├─ Metrics Enrichment (Cloud Monitoring)            │   │
│  │  ├─ Cost Estimation (Billing API)                    │   │
│  │  └─ Relationship Detection                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Platform APIs                      │
│  • Compute Engine  • Cloud Storage  • GKE                   │
│  • Cloud Monitoring  • Billing API  • Resource Manager      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
vibe-devops/
├── vivify/                    # Frontend React application
│   ├── components/            # React components
│   │   ├── gcp/              # GCP dashboard components
│   │   ├── ChatPanel.tsx     # AI chat interface
│   │   └── KanbanBoard.tsx   # Task board
│   ├── context/              # React context providers
│   ├── hooks/                # Custom React hooks
│   ├── pages/                # Page components
│   ├── services/             # API services
│   │   ├── geminiService.ts  # Gemini AI integration
│   │   ├── chatApi.ts        # Chat API client
│   │   └── gcpApi.ts         # GCP API client
│   └── types/                # TypeScript definitions
│
├── vivify-backend/           # Backend FastAPI application
│   ├── api/                  # API routes and models
│   │   ├── routes/
│   │   │   ├── chat.py       # Chat endpoints
│   │   │   └── gcp.py        # GCP discovery endpoints
│   │   └── models/           # Pydantic models
│   ├── services/             # Business logic
│   │   ├── agent_service.py  # LangChain agent
│   │   ├── gcp_discovery.py  # GCP resource discovery
│   │   ├── task_store.py     # Task management
│   │   └── tools/            # Agent tools
│   │       ├── task_tool.py
│   │       ├── canvas_tool.py
│   │       └── web_search_tool.py
│   └── utils/                # Utilities
│
└── docs-info/                # Documentation and specs
```

## Functionalities

### 1. AI Chat Assistant

The conversational agent understands natural language and can:

- **Task Management**: Create, update, list, and query tasks
  - "Show me all tasks in progress"
  - "What are the details of task-1?"
  - "List my todo tasks"

- **Infrastructure Queries**: Query GCP resources and architecture
  - "What resources are in us-central1?"
  - "Show me all compute instances"
  - "What's the total cost of my infrastructure?"

- **Web Search**: Find DevOps information and best practices
  - "What is Kubernetes?"
  - "How do I set up CI/CD?"
  - "Explain terraform modules"

- **Multi-turn Conversations**: Maintains context across messages
- **Tool Execution**: Transparently uses tools and shows progress

### 2. Kanban Task Board

Real-time task management with:

- **Drag & Drop**: Move tasks between status columns (Todo, In Progress, In Review, Deploying, Done)
- **Live Updates**: WebSocket connection with JSON Patch for instant synchronization
- **Task Details**: Click any task to view full details, subtasks, and metadata
- **Status Tracking**: Visual indicators for task status and progress
- **Automatic Reconnection**: Resilient WebSocket with exponential backoff

### 3. GCP Architecture Dashboard

Visualize your cloud infrastructure:

- **Resource Discovery**: Automatic scanning of GCP projects
  - Compute Engine VMs
  - Cloud Storage buckets
  - GKE clusters
  - VPC networks
  - Cloud Functions
  - Cloud SQL databases

- **Zone/Region Grouping**: Resources organized by geographic location
- **Live Metrics**: Real-time CPU, memory, network, and disk metrics
- **Cost Tracking**: Monthly cost estimates and breakdowns
- **Health Monitoring**: Resource health status (healthy, warning, critical)
- **Relationship Mapping**: Visual connections between resources
- **GCP Console Integration**: Direct links to resources in GCP Console

### 4. Service Account Management

- **Credential Upload**: Securely upload GCP service account JSON
- **Credential Validation**: Verify permissions before discovery
- **Project Selection**: Choose which GCP project to scan
- **Region Filtering**: Select specific regions for discovery

## Getting Started

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.8+ (for backend)
- **Google Gemini API Key** - [Get one here](https://aistudio.google.com/app/apikey)
- **GCP Service Account** (optional, for cloud discovery)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vibe-devops
   ```

2. **Set up the frontend**
   ```bash
   cd vivify
   npm install
   cp .env.example .env.local
   # Edit .env.local and add your GEMINI_API_KEY
   npm run dev
   ```

3. **Set up the backend** (in a new terminal)
   ```bash
   cd vivify-backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY and TAVILY_API_KEY
   python main.py
   ```

4. **Open your browser**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Configuration

### Frontend Environment Variables

Create `vivify/.env.local`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Backend Environment Variables

Create `vivify-backend/.env`:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
TAVILY_API_KEY=your_tavily_api_key_here  # For web search
PORT=8000
HOST=0.0.0.0
DEBUG=True
FRONTEND_URL=http://localhost:3000
```

### GCP Service Account Setup

For GCP discovery features:

1. Create a service account in GCP Console
2. Grant these roles:
   - Viewer (roles/viewer)
   - Compute Viewer
   - Storage Object Viewer
   - Kubernetes Engine Viewer
3. Download the JSON key file
4. Upload via the UI or configure in backend

## Technology Stack

### Frontend
- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Zustand** - State management
- **@dnd-kit** - Drag and drop
- **Lucide React** - Icons

### Backend
- **FastAPI** - Web framework
- **LangChain** - Agent framework
- **Google Gemini 2.0 Flash** - LLM
- **Tavily** - Web search
- **Google Cloud SDK** - GCP integration
- **WebSockets** - Real-time communication
- **Pydantic** - Data validation

## Development

### Frontend Development

```bash
cd vivify
npm run dev      # Start dev server
npm run build    # Build for production
npm run preview  # Preview production build
```

### Backend Development

```bash
cd vivify-backend
python main.py                    # Start server
uvicorn main:app --reload         # Start with auto-reload
python test_agent.py              # Test agent
python test_gemini_api.py         # Test Gemini connection
```

## API Endpoints

### Chat API
- `POST /api/chat/message` - Send message and get streaming response
- `DELETE /api/chat/sessions/{session_id}` - Clear session
- `GET /api/chat/health` - Check agent health

### GCP API
- `POST /api/gcp/validate-credentials` - Validate service account
- `POST /api/gcp/discover` - Discover GCP resources
- `GET /api/gcp/architecture/{project}` - Get cached architecture
- `DELETE /api/gcp/architecture/{project}` - Clear cache

### Task API (Coming Soon)
- `GET /api/tasks/stream/ws` - WebSocket for real-time task updates
- `GET /api/tasks` - List all tasks
- `POST /api/tasks` - Create task
- `PATCH /api/tasks/{id}` - Update task
- `DELETE /api/tasks/{id}` - Delete task

## License

MIT

---

Built with ❤️ for DevOps teams
