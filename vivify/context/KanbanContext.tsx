
import React, { createContext, useContext, useMemo, useCallback } from 'react';
import { useTaskWebSocket } from '../hooks/useTaskWebSocket';
import { Task, TaskStatus } from '../types';
import { COLUMN_ORDER } from '../constants';

type GroupedTasks = Record<TaskStatus, Task[]>;

interface KanbanContextType {
  tasksById: Record<string, Task>;
  groupedTasks: GroupedTasks;
  tasks: Task[];
  isLoading: boolean;
  isConnected: boolean;
  error: Error | null;
  updateTaskStatus: (taskId: string, newStatus: TaskStatus) => void;
  setSelectedTask: (task: Task | null) => void;
}

const KanbanContext = createContext<KanbanContextType | undefined>(undefined);

export const KanbanProvider: React.FC<{
  children: React.ReactNode;
  setSelectedTask: (task: Task | null) => void;
}> = ({ children, setSelectedTask }) => {
  const { tasks: tasksById, isLoading, isConnected, error, updateTaskStatus: wsUpdateTaskStatus } = useTaskWebSocket('mock_project_id');

  const tasks = useMemo(() => {
    return Object.values(tasksById).sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }, [tasksById]);

  const groupedTasks = useMemo(() => {
    const initialGroups: GroupedTasks = {} as GroupedTasks;
    COLUMN_ORDER.forEach(status => {
      initialGroups[status] = [];
    });

    return tasks.reduce((acc, task) => {
      if (acc[task.status]) {
        acc[task.status].push(task);
      }
      return acc;
    }, initialGroups);
  }, [tasks]);
  
  const updateTaskStatus = useCallback((taskId: string, newStatus: TaskStatus) => {
    // This would typically be a REST API call.
    // The WebSocket server would then broadcast the change.
    // For this demo, we'll directly update via the mock WebSocket.
    console.log(`Updating task ${taskId} to status ${newStatus}`);
    wsUpdateTaskStatus(taskId, newStatus);
  }, [wsUpdateTaskStatus]);


  const value = {
    tasksById,
    groupedTasks,
    tasks,
    isLoading,
    isConnected,
    error,
    updateTaskStatus,
    setSelectedTask,
  };

  return <KanbanContext.Provider value={value}>{children}</KanbanContext.Provider>;
};

export const useKanban = (): KanbanContextType => {
  const context = useContext(KanbanContext);
  if (context === undefined) {
    throw new Error('useKanban must be used within a KanbanProvider');
  }
  return context;
};
