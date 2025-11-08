
import { TaskStatus } from './types';

export const COLUMN_ORDER: TaskStatus[] = [
  TaskStatus.Todo,
  TaskStatus.InProgress,
  TaskStatus.InReview,
  TaskStatus.Done,
  TaskStatus.Cancelled,
];

export const STATUS_CONFIG: Record<TaskStatus, { label: string; color: string }> = {
  [TaskStatus.Todo]: {
    label: 'To Do',
    color: 'bg-gray-500',
  },
  [TaskStatus.InProgress]: {
    label: 'In Progress',
    color: 'bg-blue-500',
  },
  [TaskStatus.InReview]: {
    label: 'In Review',
    color: 'bg-yellow-500',
  },
  [TaskStatus.Done]: {
    label: 'Done',
    color: 'bg-green-500',
  },
  [TaskStatus.Cancelled]: {
    label: 'Cancelled',
    color: 'bg-red-500',
  },
};
