import React, { useState, useRef, useEffect } from 'react';
import { chatApi } from '../services/chatApi';
import { PaperAirplaneIcon } from './icons/PaperAirplaneIcon';
import { ChatMessage, ModelStep } from '../types';
import { CogIcon } from './icons/CogIcon';
import { TerminalIcon } from './icons/TerminalIcon';
import { produce } from 'immer';
import { useGCPConnection } from '../context/GCPConnectionContext';
import { RadarIcon } from './icons/RadarIcon';

const CHAT_STORAGE_KEY = 'vibe_devops_chat_history';

const initialMessage: ChatMessage = {
  id: 'init-1',
  role: 'model',
  text: 'Hello! I am Vibe, your DevOps assistant. I can help you list tasks (e.g., "list my inprogress tasks"), query your GCP resources, or answer questions. How can I help?',
  timestamp: new Date().toISOString(),
};

const TOOL_ICONS: Record<string, React.FC<React.SVGProps<SVGSVGElement>>> = {
  discover_gcp_resources: RadarIcon,
  task_manager: TerminalIcon,
  canvas_query: RadarIcon,
  web_search: TerminalIcon,
};


const ThinkingBubble: React.FC<{ thought: string }> = ({ thought }) => (
  <div className="flex items-start space-x-2 text-xs text-gray-400 my-2">
    <CogIcon className="w-4 h-4 flex-shrink-0 animate-spin" />
    <p className="italic">{thought}</p>
  </div>
);

const ToolCallBubble: React.FC<{ toolName: string; toolInput: string }> = ({ toolName, toolInput }) => {
  const ToolIcon = TOOL_ICONS[toolName] || TerminalIcon;
  return (
    <div className="my-2 text-xs text-gray-400">
      <div className="flex items-center space-x-2">
          <ToolIcon className="w-4 h-4 flex-shrink-0" />
          <p>Using tool: <span className="font-semibold text-gray-300">{toolName}</span></p>
      </div>
      <pre className="mt-1 p-2 bg-gray-900 rounded-md text-gray-300 font-mono text-[11px] max-h-32 overflow-auto">{toolInput}</pre>
    </div>
  );
};

const ModelMessage: React.FC<{ message: ChatMessage }> = ({ message }) => (
  <div className="flex justify-start">
    <div className="max-w-xs md:max-w-sm lg:max-w-lg p-3 rounded-lg bg-gray-700 text-gray-200">
      {message.steps?.map((step, i) => {
        if (step.type === 'thinking') return <ThinkingBubble key={i} thought={step.thought} />;
        if (step.type === 'tool_call') return <ToolCallBubble key={i} toolName={step.toolName} toolInput={step.toolInput} />;
        // Tool results are often noisy, so we don't display them directly. The final answer summarizes them.
        return null;
      })}
      {message.text && (
        <p className="text-sm" style={{ whiteSpace: 'pre-wrap' }}>{message.text}</p>
      )}
      {!message.text && !message.steps?.some(s => s.type === 'final_answer') && (
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-75"></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-150"></div>
        </div>
      )}
    </div>
  </div>
);


const ChatPanel: React.FC = () => {
  const { hasGCPAccess } = useGCPConnection();
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const storedMessages = localStorage.getItem(CHAT_STORAGE_KEY);
      return storedMessages ? JSON.parse(storedMessages) : [initialMessage];
    } catch (error) {
      console.error("Failed to parse chat history:", error);
      return [initialMessage];
    }
  });

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: input,
      timestamp: new Date().toISOString(),
    };
    
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    const modelMessageId = `model-${Date.now()}`;
    const sessionId = `session-${Date.now()}`;  // Generate session ID
    
    setMessages(currentMessages => [...currentMessages, { 
      id: modelMessageId, 
      role: 'model', 
      steps: [], 
      timestamp: new Date().toISOString() 
    }]);

    try {
      await chatApi.streamMessage(
        {
          message: input,
          session_id: sessionId,
          history: []  // History is managed by backend
        },
        (chunk: ModelStep) => {
          setMessages(currentMessages => produce(currentMessages, draft => {
            const messageToUpdate = draft.find(m => m.id === modelMessageId);
            if (messageToUpdate) {
              if (!messageToUpdate.steps) messageToUpdate.steps = [];
              messageToUpdate.steps.push(chunk);
              if (chunk.type === 'final_answer') {
                messageToUpdate.text = chunk.text;
              }
            }
          }));
        }
      );
    } catch (error) {
      console.error('Error streaming chat response:', error);
      setMessages(currentMessages => produce(currentMessages, draft => {
          const messageToUpdate = draft.find(m => m.id === modelMessageId);
          if (messageToUpdate) {
            messageToUpdate.text = `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`;
          }
      }));
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatTimestamp = (isoString: string) => {
    return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  return (
    <div className="flex flex-col h-full bg-gray-800 text-gray-100">
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              {msg.role === 'user' ? (
                <div className="max-w-xs md:max-w-sm lg:max-w-md p-3 rounded-lg bg-blue-600 text-white">
                    <p className="text-sm" style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</p>
                </div>
              ) : (
                <ModelMessage message={msg} />
              )}
               <span className="text-xs text-gray-500 mt-1 px-1">{formatTimestamp(msg.timestamp)}</span>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center bg-gray-700 rounded-lg">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask Vibe anything..."
            className="flex-1 p-3 bg-transparent border-none rounded-lg focus:ring-0 text-sm"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="p-3 text-gray-400 hover:text-blue-500 disabled:text-gray-600 transition-colors"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
        <div className="text-xs text-center text-gray-500 pt-2">
          Agent: <span className="text-green-400">Backend Connected</span>
          <span className="mx-2">|</span>
          GCP: {hasGCPAccess ? (
            <span className="text-green-400">Connected</span>
          ) : (
            <span className="text-yellow-400">Not Connected</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;