import React, { useEffect, useCallback } from 'react';
import { Task, Subtask, TaskStatus } from '../types';
import { STATUS_CONFIG } from '../constants';
import { XIcon } from './icons/XIcon';

interface TaskDetailsModalProps {
  task: Task;
  onClose: () => void;
}

const TaskDetailsModal: React.FC<TaskDetailsModalProps> = ({ task, onClose }) => {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      onClose();
    }
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
  
  const statusConfig = STATUS_CONFIG[task.status as TaskStatus];

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-70 flex justify-center items-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <span className={`w-4 h-4 rounded-full ${statusConfig.color}`}></span>
            <h2 className="text-xl font-bold text-white">{task.title}</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            <XIcon className="w-6 h-6" />
          </button>
        </header>

        <div className="flex-1 p-6 overflow-y-auto">
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Description</h3>
              <p className="text-gray-300 whitespace-pre-wrap">{task.description || 'No description available.'}</p>
            </div>
            
            {task.subtasks && task.subtasks.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-3">Subtasks</h3>
                <ul className="space-y-2">
                  {task.subtasks.map((subtask: Subtask) => (
                    <li key={subtask.id} className="flex items-center bg-gray-700 p-2 rounded">
                      <input type="checkbox" checked={subtask.status === 'completed'} readOnly className="mr-3 h-4 w-4 rounded bg-gray-600 border-gray-500 text-blue-500 focus:ring-blue-500" />
                      <span className={`${subtask.status === 'completed' ? 'line-through text-gray-500' : 'text-gray-200'}`}>{subtask.title}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {task.metadata && (
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-2">Metadata</h3>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm bg-gray-700 p-3 rounded-md">
                  {Object.entries(task.metadata).map(([key, value]) => (
                    <div key={key} className="flex flex-col">
                      <span className="capitalize text-gray-400">{key.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-white">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Details</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="text-gray-400">Status: <span className="font-medium text-white">{statusConfig.label}</span></div>
                <div className="text-gray-400">Created: <span className="font-medium text-white">{new Date(task.created_at).toLocaleString()}</span></div>
                <div className="text-gray-400">Last Updated: <span className="font-medium text-white">{new Date(task.updated_at).toLocaleString()}</span></div>
                <div className="text-gray-400">Task ID: <span className="font-mono text-xs text-gray-500">{task.id}</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TaskDetailsModal;
