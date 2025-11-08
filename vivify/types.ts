
export enum TaskStatus {
  Todo = 'todo',
  InProgress = 'inprogress',
  InReview = 'inreview',
  Done = 'done',
  Cancelled = 'cancelled',
}

export interface Subtask {
  id: string;
  title: string;
  description?: string;
  status: 'pending' | 'completed';
  task_id: string;
}

export interface Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  created_at: string;
  updated_at: string;
  subtasks?: Subtask[];
  metadata?: Record<string, any>;
}

export interface TasksState {
  tasks: Record<string, Task>;
}

export interface JsonPatchOperation {
  op: 'add' | 'replace' | 'remove';
  path: string;
  value?: any;
}

// New Types for Advanced Chat

export interface ThinkingStep {
  type: 'thinking';
  thought: string;
}
export interface ToolCallStep {
  type: 'tool_call';
  toolName: string;
  toolInput: string;
}
export interface ToolResultStep {
    type: 'tool_result';
    toolName: string;
    toolOutput: string;
}
export interface FinalAnswerStep {
  type: 'final_answer';
  text: string;
}

export type ModelStep = ThinkingStep | ToolCallStep | ToolResultStep | FinalAnswerStep;

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  // User messages have text, model messages build up steps and then get a final text value
  text?: string; 
  steps?: ModelStep[];
  timestamp: string;
}
