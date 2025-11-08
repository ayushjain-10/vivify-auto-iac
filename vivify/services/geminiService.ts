import { GoogleGenAI } from "@google/genai";
import { ChatMessage, ModelStep, Task } from '../types';
import { MOCK_INITIAL_TASKS } from '../hooks/useTaskWebSocket';

const API_KEY = process.env.GEMINI_API_KEY || process.env.API_KEY;
const ai = API_KEY ? new GoogleGenAI({ apiKey: API_KEY }) : null;
const model = 'gemini-2.0-flash-exp';

// --- Tool Simulation ---
// In a real backend, these would be actual functions that interact with databases, etc.

const tools = {
  query_tasks: (args: { status?: string, title_contains?: string }) => {
    console.log("TOOL: query_tasks called with", args);
    let tasks = Object.values(MOCK_INITIAL_TASKS);
    if (args.status) {
      tasks = tasks.filter(t => t.status === args.status);
    }
    if (args.title_contains) {
      tasks = tasks.filter(t => t.title.toLowerCase().includes(args.title_contains!.toLowerCase()));
    }
    return tasks.map(t => ({ id: t.id, title: t.title, status: t.status }));
  },
  get_task_details: (args: { id?: string; title?: string }) => {
    console.log("TOOL: get_task_details called with", args);
    const tasks = Object.values(MOCK_INITIAL_TASKS);
    let foundTask: Task | undefined;
    if (args.id) {
      foundTask = tasks.find(t => t.id === args.id);
    } else if (args.title) {
      const searchTerm = args.title.toLowerCase().trim();
      foundTask = tasks.find(t => t.title.toLowerCase().includes(searchTerm));
    }
    return foundTask || { error: "Task not found." };
  },
  discover_gcp_resources: (args: { project: string }) => {
    console.log("TOOL: discover_gcp_resources called with", args);
    // This would trigger a backend job. We'll just return a confirmation.
    // In a more advanced implementation, this could use an event bus to notify the frontend to start refreshing.
    return `Discovery process initiated for project '${args.project}'. The architecture canvas will be updated with the latest data. You can navigate to the canvas to see the results.`;
  },
  web_search: (args: { query: string }) => {
    console.log("TOOL: web_search called with", args);
    if (args.query.toLowerCase().includes('terraform')) {
      return "Terraform is an open-source infrastructure as code software tool created by HashiCorp. It enables users to define and provision a data center infrastructure using a high-level configuration language known as Hashicorp Configuration Language (HCL), or optionally JSON.";
    }
    return "No relevant information found for your query.";
  }
};

// --- Agent Simulation ---

const runAgentSimulation = (prompt: string, onChunk: (chunk: ModelStep) => void) => {
  const lowerCasePrompt = prompt.toLowerCase();

  // Scenario: Discover GCP Resources
  if (lowerCasePrompt.includes('discover') || lowerCasePrompt.includes('scan') || lowerCasePrompt.includes('refresh canvas') || lowerCasePrompt.includes('find my resources')) {
    const toolInput = { project: 'vibe-devops-project' }; // This would be dynamically determined in a real app
    setTimeout(() => onChunk({ type: 'thinking', thought: 'The user wants to see their GCP resources. I should use the `discover_gcp_resources` tool to start a scan.' }), 500);
    setTimeout(() => onChunk({ type: 'tool_call', toolName: 'discover_gcp_resources', toolInput: JSON.stringify(toolInput, null, 2) }), 1200);
    setTimeout(() => {
        const result = tools.discover_gcp_resources(toolInput);
        onChunk({ type: 'tool_result', toolName: 'discover_gcp_resources', toolOutput: result });
        setTimeout(() => onChunk({ type: 'final_answer', text: result }), 1000);
    }, 2200);
    return;
  }


  // Scenario: Get details for a specific task
  const detailMatch = lowerCasePrompt.match(/(?:details for|tell me about|more on|show details for) (?:task-)?(.+)/);
  if (detailMatch) {
    const identifier = detailMatch[1].replace(/['"]/g, '').trim();
    // A simple way to check if it's likely an ID or a title
    const isIdLike = /^\d+$/.test(identifier) || MOCK_INITIAL_TASKS[`task-${identifier}`];
    const toolInput = isIdLike ? { id: `task-${identifier}` } : { title: identifier };
    
    setTimeout(() => onChunk({ type: 'thinking', thought: 'The user wants details for a specific task. I should use the `get_task_details` tool.' }), 500);
    setTimeout(() => onChunk({ type: 'tool_call', toolName: 'get_task_details', toolInput: JSON.stringify(toolInput, null, 2) }), 1200);

    setTimeout(() => {
        const result = tools.get_task_details(toolInput);
        onChunk({ type: 'tool_result', toolName: 'get_task_details', toolOutput: JSON.stringify(result, null, 2) });
        
        setTimeout(() => onChunk({ type: 'thinking', thought: 'I have the task details. I will now format them into a comprehensive summary.' }), 500);

        setTimeout(() => {
            let answerText;
            // FIX: Use a type guard to correctly handle the union type of `result`.
            if (result && !('error' in result)) {
                answerText = `**Details for "${result.title}" (ID: ${result.id})**\n\n`;
                answerText += `* **Status:** ${result.status}\n`;
                answerText += `* **Description:** ${result.description || 'N/A'}\n\n`;
                
                if (result.subtasks && result.subtasks.length > 0) {
                    answerText += `**Subtasks:**\n`;
                    answerText += result.subtasks.map(st => `  - [${st.status === 'completed' ? 'x' : ' '}] ${st.title}`).join('\n') + '\n\n';
                }

                if (result.metadata) {
                    answerText += `**Metadata:**\n`;
                    answerText += Object.entries(result.metadata).map(([key, value]) => `  - **${key.charAt(0).toUpperCase() + key.slice(1)}:** ${value}`).join('\n');
                }

            } else {
                answerText = `I'm sorry, I couldn't find a task matching "${identifier}".`;
            }
            onChunk({ type: 'final_answer', text: answerText.trim() });
        }, 1500);

    }, 2200);
    return;
  }

  // Scenario: User asks to list tasks
  if (lowerCasePrompt.includes('list') && lowerCasePrompt.includes('tasks')) {
    const statusMatch = lowerCasePrompt.match(/(\w+)\s+tasks/);
    const status = statusMatch ? statusMatch[1] : undefined;
    const toolInput = { status };

    setTimeout(() => onChunk({ type: 'thinking', thought: 'The user wants to see tasks. I should use the `query_tasks` tool.' }), 500);
    setTimeout(() => onChunk({ type: 'tool_call', toolName: 'query_tasks', toolInput: JSON.stringify(toolInput, null, 2) }), 1200);
    
    setTimeout(() => {
      const results = tools.query_tasks(toolInput);
      onChunk({ type: 'tool_result', toolName: 'query_tasks', toolOutput: JSON.stringify(results, null, 2) });

      setTimeout(() => onChunk({ type: 'thinking', thought: 'I have the task list. Now I will format it as a clear, human-readable answer.' }), 500);
      
      setTimeout(() => {
        const answerText = results.length > 0
          ? `I found ${results.length} task(s):\n${results.map(t => `- **${t.title}** (Status: ${t.status})`).join('\n')}`
          : "I couldn't find any tasks matching your criteria.";
        onChunk({ type: 'final_answer', text: answerText });
      }, 1500);

    }, 2200);
    return;
  }

  // Scenario: User asks a general question (simulating web search)
  if (lowerCasePrompt.includes('what is') || lowerCasePrompt.includes('who is')) {
     const query = prompt.replace(/what is|who is/i, '').trim();
     const toolInput = { query };
     setTimeout(() => onChunk({ type: 'thinking', thought: 'This is a general knowledge question. I should use the `web_search` tool.' }), 500);
     setTimeout(() => onChunk({ type: 'tool_call', toolName: 'web_search', toolInput: JSON.stringify(toolInput, null, 2) }), 1200);
     setTimeout(() => {
        const result = tools.web_search(toolInput);
        onChunk({ type: 'tool_result', toolName: 'web_search', toolOutput: result});
        setTimeout(() => onChunk({ type: 'final_answer', text: result }), 1000);
     }, 2200);
     return;
  }
  
  // Default Scenario: Simple text response
  setTimeout(() => onChunk({ type: 'thinking', thought: 'This is a straightforward request. I will provide a direct answer.' }), 500);
  setTimeout(() => onChunk({ type: 'final_answer', text: `I'm sorry, I'm not equipped to handle that request yet. I am a demo assistant and can currently help with listing tasks (e.g., "list my todo tasks") or general questions like "What is Terraform?".` }), 1500);
};


export const streamGeminiResponse = async (
  prompt: string,
  history: ChatMessage[],
  onChunk: (chunk: ModelStep) => void
): Promise<void> => {
  if (!API_KEY || !ai) {
    onChunk({ type: 'final_answer', text: "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to the .env.local file." });
    return;
  }

  // Check if we should use simulation for specific commands
  const lowerPrompt = prompt.toLowerCase();
  const useSimulation = lowerPrompt.includes('list') && lowerPrompt.includes('tasks') ||
                        lowerPrompt.includes('details for') ||
                        lowerPrompt.includes('discover') ||
                        lowerPrompt.includes('scan');

  if (useSimulation) {
    console.log("Using simulation for task/GCP commands:", prompt);
    runAgentSimulation(prompt, onChunk);
    return;
  }

  // Real Gemini API call for general questions
  console.log("Calling Gemini API for prompt:", prompt);
  
  try {
    // Convert chat history to Gemini format
    const geminiHistory = history
      .filter(msg => msg.text) // Only include messages with text
      .map(msg => ({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.text || '' }]
      }));

    const chat = ai.models.generateContent({
      model,
      contents: [
        ...geminiHistory,
        { role: 'user', parts: [{ text: prompt }] }
      ],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2048,
      }
    });

    onChunk({ type: 'thinking', thought: 'Processing your request...' });

    const result = await chat;
    const response = result.response;
    const text = response.text();

    onChunk({ type: 'final_answer', text });

  } catch (error) {
    console.error('Error calling Gemini API:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    onChunk({ type: 'final_answer', text: `Sorry, I encountered an error: ${errorMessage}` });
  }
};