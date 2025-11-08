
import React from 'react';
import {
  DndContext,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  closestCenter,
} from '@dnd-kit/core';
import { useKanban } from '../context/KanbanContext';
import KanbanColumn from './KanbanColumn';
import { Task, TaskStatus } from '../types';
import { COLUMN_ORDER } from '../constants';

interface KanbanBoardProps {
  onCardClick: (task: Task) => void;
}

const KanbanBoard: React.FC<KanbanBoardProps> = ({ onCardClick }) => {
  const { groupedTasks, isLoading, updateTaskStatus } = useKanban();

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const taskId = active.id as string;
      const newStatus = over.id as TaskStatus;
      
      const currentStatus = (active.data.current?.task as Task)?.status;

      if (currentStatus && newStatus !== currentStatus) {
         updateTaskStatus(taskId, newStatus);
      }
    }
  };

  if (isLoading) {
    return <div className="flex items-center justify-center h-full"><p>Loading tasks...</p></div>;
  }

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      <div className="flex h-full p-4 space-x-4 overflow-x-auto bg-gray-900">
        {COLUMN_ORDER.map((status) => (
          <KanbanColumn
            key={status}
            status={status}
            tasks={groupedTasks[status] || []}
            onCardClick={onCardClick}
          />
        ))}
      </div>
    </DndContext>
  );
};

export default KanbanBoard;
