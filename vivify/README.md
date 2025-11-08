<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Vibe DevOps - Cloud Architecture & Task Management Platform

A comprehensive DevOps platform combining AI-powered task management with real-time GCP infrastructure visualization.

## Features

- 🤖 **AI Chat Assistant** - Gemini-powered conversational agent for task management
- 📋 **Kanban Board** - Real-time task tracking with drag-and-drop
- ☁️ **GCP Architecture Dashboard** - Visualize your cloud infrastructure
- 🔄 **WebSocket Integration** - Live updates using JSON Patch
- 🎨 **Modern UI** - Dark theme with responsive design

## Prerequisites

- **Node.js** (v18 or higher)
- **Gemini API Key** - Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Quick Start

### Option 1: Automated Setup (Recommended)

**On macOS/Linux:**
```bash
./setup.sh
```

**On Windows:**
```bash
setup.bat
```

Then edit `.env.local` and add your Gemini API key, and run:
```bash
npm run dev
```

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure your API key:**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env.local
   ```
   
   Then edit `.env.local` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   
   Navigate to [http://localhost:3000](http://localhost:3000)

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key for the chat assistant | Yes |

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into your `.env.local` file

## Usage

### Chat Assistant

The chat panel supports:
- **Task queries**: "list my todo tasks", "show inprogress tasks"
- **Task details**: "details for task-1", "tell me about Setup CI/CD"
- **GCP discovery**: "scan my project", "discover resources"
- **General questions**: "what is terraform?", "explain kubernetes"

### Kanban Board

- Drag and drop tasks between columns
- Click on any task card to view details
- Real-time updates via WebSocket simulation

### GCP Architecture Dashboard

- View your cloud resources grouped by zone/region
- Click on resources to see detailed information
- Monitor costs and health status
- (Note: Currently uses mock data - backend integration coming soon)

## Project Structure

```
vivify/
├── components/       # React components
│   ├── gcp/         # GCP dashboard components
│   └── icons/       # Icon components
├── context/         # React context providers
├── hooks/           # Custom React hooks
├── pages/           # Page components
├── services/        # API services (Gemini)
├── types/           # TypeScript type definitions
├── utils/           # Utility functions
└── mock/            # Mock data for development
```

## Development

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Troubleshooting

### Chat not working?

- Make sure your `GEMINI_API_KEY` is set in `.env.local`
- Check the browser console for error messages
- Verify your API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)

### Tasks not updating?

- The WebSocket connection is currently simulated with mock data
- Backend integration is required for real-time task updates

## Roadmap

- [ ] Backend integration with Python/FastAPI
- [ ] Real WebSocket server for tasks
- [ ] VivifyRT integration for GCP discovery
- [ ] Network topology visualization
- [ ] Cost analysis and optimization
- [ ] Multi-cloud support (AWS, Azure)

## License

MIT
