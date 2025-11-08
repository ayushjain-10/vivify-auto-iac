
import React from 'react';
import { useDroppable } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import TaskCard from './TaskCard';
import { Task, TaskStatus } from '../types';
import { STATUS_CONFIG } from '../constants';

interface KanbanColumnProps {
  status: TaskStatus;
  tasks: Task[];
  onCardClick: (task: Task) => void;
}

const KanbanColumn: React.FC<KanbanColumnProps> = ({ status, tasks, onCardClick }) => {
  const { setNodeRef, isOver } = useDroppable({ id: status });
  const { label, color } = STATUS_CONFIG[status];

  return (
    <div
      ref={setNodeRef}
      className={`flex flex-col w-72 md:w-80 lg:w-96 flex-shrink-0 rounded-lg bg-gray-800 shadow-lg h-full transition-colors duration-200 ${isOver ? 'bg-gray-700' : ''}`}
    >
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <span className={`w-3 h-3 rounded-full ${color}`}></span>
          <h2 className="font-bold text-white">{label}</h2>
        </div>
        <span className="text-sm font-medium text-gray-400 bg-gray-700 rounded-full px-2 py-1">
          {tasks.length}
        </span>
      </div>
      <div className="flex-1 p-2 overflow-y-auto space-y-2">
        <SortableContext items={tasks.map(t => t.id)} strategy={verticalListSortingStrategy}>
          {tasks.map((task) => (
            <TaskCard key={task.id} task={task} onClick={() => onCardClick(task)} />
          ))}
        </SortableContext>
      </div>
    </div>
  );
};

export default KanbanColumn;
