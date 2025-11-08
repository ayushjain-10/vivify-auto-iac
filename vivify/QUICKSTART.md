# Quick Start Guide

Get Vibe DevOps running in 3 minutes! ⚡

## Step 1: Install Dependencies (30 seconds)

```bash
npm install
```

## Step 2: Get Your Gemini API Key (1 minute)

1. Open [Google AI Studio](https://aistudio.google.com/app/apikey) in your browser
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (it looks like: `AIzaSyC...`)

## Step 3: Configure the API Key (30 seconds)

Create a file named `.env.local` in the `vivify` folder:

```bash
# On macOS/Linux:
cp .env.example .env.local

# On Windows:
copy .env.example .env.local
```

Open `.env.local` and paste your API key:

```
GEMINI_API_KEY=AIzaSyC...your_actual_key_here
```

## Step 4: Start the App (10 seconds)

```bash
npm run dev
```

## Step 5: Open in Browser

Navigate to: **http://localhost:3000**

## What You'll See

### Cloud DevOps Architect Tab
- **Left Panel**: AI chat assistant (try: "What is Kubernetes?")
- **Right Panel**: Kanban board with draggable task cards

### Live Architecture Canvas Tab
- GCP resource visualization (currently using mock data)
- Click "Connect to GCP" to add your service account

## Try These Commands

In the chat panel, try:

```
list my todo tasks
```

```
details for task-1
```

```
what is terraform?
```

```
scan my project
```

## Troubleshooting

### Chat shows "API key not configured"
- Make sure `.env.local` exists in the `vivify` folder
- Check that you pasted the full API key
- Restart the dev server: Stop (Ctrl+C) and run `npm run dev` again

### "Cannot find module" errors
- Run `npm install` again
- Delete `node_modules` and run `npm install`

### Port 3000 already in use
- Stop other apps using port 3000
- Or change the port in `vite.config.ts`

## Next Steps

- Read [CONFIGURATION.md](./CONFIGURATION.md) for advanced setup
- Check [README.md](./README.md) for full documentation
- Explore the chat commands in [docs-info/chat-commands.md](./docs-info/chat-commands.md)

## Need Help?

- Check the browser console (F12) for errors
- Verify your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Make sure you're using Node.js v18 or higher: `node --version`

---

**That's it!** You're ready to use Vibe DevOps. 🎉
