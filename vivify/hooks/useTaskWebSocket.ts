import { useState, useEffect, useRef, useCallback } from 'react';
import { produce } from 'immer';
import { applyPatch } from '../utils/jsonPatch';
import { Task, TaskStatus, JsonPatchOperation } from '../types';

// Mock data to simulate server responses
export const MOCK_INITIAL_TASKS: Record<string, Task> = {
  'task-1': {
    id: 'task-1',
    title: 'Setup CI/CD Pipeline',
    description: 'Configure GitHub Actions for automated testing and deployment.',
    status: TaskStatus.Todo,
    created_at: '2023-10-26T10:00:00Z',
    updated_at: '2023-10-26T10:00:00Z',
    subtasks: [
      { id: 'sub-1', title: 'Configure SonarCloud analysis', status: 'pending', task_id: 'task-1' },
      { id: 'sub-2', title: 'Set up deployment to staging', status: 'completed', task_id: 'task-1' }
    ]
  },
  'task-2': {
    id: 'task-2',
    title: 'Develop User Authentication',
    description: 'Implement JWT-based authentication for the main API.',
    status: TaskStatus.InProgress,
    created_at: '2023-10-25T11:00:00Z',
    updated_at: '2023-10-25T12:30:00Z',
    metadata: {
      assignee: 'Alice',
      priority: 'High',
      sprint: 'Sprint 3'
    }
  },
  'task-3': { id: 'task-3', title: 'Design Database Schema', description: 'Finalize the PostgreSQL schema for all services.', status: TaskStatus.Done, created_at: '2023-10-24T09:00:00Z', updated_at: '2023-10-24T15:00:00Z' },
  'task-4': { id: 'task-4', title: 'Code Review for Feature X', description: 'Review pull request #123 for the new notifications feature.', status: TaskStatus.InReview, created_at: '2023-10-26T14:00:00Z', updated_at: '2023-10-26T14:00:00Z' },
  'task-5': { id: 'task-5', title: 'Fix Login Page CSS Bug', description: 'The login button is misaligned on mobile devices.', status: TaskStatus.InProgress, created_at: '2023-10-26T15:00:00Z', updated_at: '2023-10-26T15:00:00Z' },
  'task-6': { id: 'task-6', title: 'Deploy Staging Environment', description: 'Push the latest build to the staging server for QA testing.', status: TaskStatus.Todo, created_at: '2023-10-26T16:00:00Z', updated_at: '2023-10-26T16:00:00Z' },
};

export const useTaskWebSocket = (projectId: string) => {
  // FIX: The state shape was incorrect. The JSON patches expect a root object with a `tasks` property.
  const [state, setState] = useState<{ tasks: Record<string, Task> }>({ tasks: {} });
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const retryAttempt = useRef(0);
  const mockTimeoutRef = useRef<any>(null);
  const mockIntervalRef = useRef<any>(null);

  const applyPatches = useCallback((patches: JsonPatchOperation[]) => {
    try {
      setState(currentState => {
        const nextState = produce(currentState, draft => {
          applyPatch(draft, patches);
        });
        return nextState;
      });
    } catch (e) {
      console.error('Failed to apply patch:', e);
      setError(e instanceof Error ? e : new Error('Failed to apply patch'));
    }
  }, []);

  const connect = useCallback(() => {
    // This simulates a WebSocket connection.
    console.log(`Connecting to ws://localhost:8080/api/tasks/stream/ws?project_id=${projectId}`);
    setIsConnected(false);

    mockTimeoutRef.current = setTimeout(() => {
      // onopen
      console.log('WebSocket connection established.');
      setIsConnected(true);
      setIsLoading(false);
      retryAttempt.current = 0;
      setError(null);

      // onmessage - initial snapshot
      const initialPatch: JsonPatchOperation[] = [{ op: 'replace', path: '/tasks', value: MOCK_INITIAL_TASKS }];
      applyPatches(initialPatch);

      // Note: Removed the interval that was adding tasks every 8 seconds
      // Tasks are now static and only updated via updateTaskStatus

    }, 1000); // Simulate connection delay
  }, [projectId, applyPatches]);

  useEffect(() => {
    connect();
    return () => {
      // Cleanup: clear both timeout and interval
      if (mockTimeoutRef.current) {
        clearTimeout(mockTimeoutRef.current);
      }
      if (mockIntervalRef.current) {
        clearInterval(mockIntervalRef.current);
      }
    };
  }, [connect]);
  
  const updateTaskStatus = useCallback((taskId: string, newStatus: TaskStatus) => {
    // This function simulates the client sending an update and receiving a patch back.
    const patch: JsonPatchOperation[] = [
      { op: 'replace', path: `/tasks/${taskId}/status`, value: newStatus },
      { op: 'replace', path: `/tasks/${taskId}/updated_at`, value: new Date().toISOString() }
    ];
    applyPatches(patch);
  }, [applyPatches]);

  return { tasks: state.tasks, isLoading, isConnected, error, updateTaskStatus };
};
