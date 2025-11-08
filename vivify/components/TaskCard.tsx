
import React from 'react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { Task, TaskStatus } from '../types';
import { STATUS_CONFIG } from '../constants';

interface TaskCardProps {
  task: Task;
  onClick: () => void;
}

const TaskCard: React.FC<TaskCardProps> = ({ task, onClick }) => {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: task.id,
    data: { task },
  });

  const style = {
    transform: CSS.Translate.toString(transform),
    zIndex: isDragging ? 100 : 'auto',
    opacity: isDragging ? 0.8 : 1,
  };
  
  const statusColor = STATUS_CONFIG[task.status as TaskStatus]?.color || 'bg-gray-500';

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
      onClick={onClick}
      className="p-4 bg-gray-700 rounded-lg shadow-md cursor-pointer hover:bg-gray-600 transition-all duration-200 border-l-4 border-transparent hover:border-blue-500"
    >
        <div className="flex justify-between items-start">
            <h3 className="font-bold text-white mb-2">{task.title}</h3>
            <span className={`w-2.5 h-2.5 rounded-full ${statusColor} mt-1 flex-shrink-0`}></span>
        </div>
      
      <p className="text-sm text-gray-400 line-clamp-3">
        {task.description || 'No description provided.'}
      </p>
      <div className="text-xs text-gray-500 mt-3 pt-2 border-t border-gray-600">
        Updated: {new Date(task.updated_at).toLocaleDateString()}
      </div>
    </div>
  );
};

export default TaskCard;
