import React, { useState, useRef, useEffect, useCallback } from 'react';
import ChatPanel from '../components/ChatPanel';
import KanbanBoard from '../components/KanbanBoard';
import { KanbanProvider } from '../context/KanbanContext';
import TaskDetailsModal from '../components/TaskDetailsModal';
import { Task } from '../types';

const MIN_PANEL_WIDTH = 288; // Corresponds to w-72
const MAX_PANEL_WIDTH = 640; // Arbitrary max width for user comfort
const STORAGE_KEY = 'vibe_devops_panel_width';
const DEFAULT_WIDTH = 320; // Corresponds to w-80

const CloudDevopsArchitectPage: React.FC = () => {
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    try {
      const storedWidth = localStorage.getItem(STORAGE_KEY);
      const parsedWidth = storedWidth ? parseInt(storedWidth, 10) : DEFAULT_WIDTH;
      // Ensure the stored width is within bounds
      return Math.max(MIN_PANEL_WIDTH, Math.min(parsedWidth, MAX_PANEL_WIDTH));
    } catch {
      return DEFAULT_WIDTH;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, panelWidth.toString());
    } catch (error) {
      console.error("Failed to save panel width to localStorage", error);
    }
  }, [panelWidth]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    const handleMouseMove = (event: MouseEvent) => {
      let newWidth = event.clientX;
      
      if (newWidth < MIN_PANEL_WIDTH) newWidth = MIN_PANEL_WIDTH;
      if (newWidth > MAX_PANEL_WIDTH) newWidth = MAX_PANEL_WIDTH;
      
      setPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

  }, []);

  return (
    <KanbanProvider setSelectedTask={setSelectedTask}>
      <div className="flex h-full">
        <div
          style={{ width: `${panelWidth}px` }}
          className="hidden md:flex flex-col flex-shrink-0 bg-gray-800"
        >
          <ChatPanel />
        </div>
        
        <div
          onMouseDown={handleMouseDown}
          className="hidden md:block w-1.5 cursor-col-resize flex-shrink-0 bg-gray-700 hover:bg-blue-600 transition-colors duration-200"
          aria-label="Resize panel"
          role="separator"
        />

        <main className="flex-1 overflow-x-auto overflow-y-hidden">
          <KanbanBoard onCardClick={setSelectedTask} />
        </main>
      </div>
      {selectedTask && (
        <TaskDetailsModal task={selectedTask} onClose={() => setSelectedTask(null)} />
      )}
    </KanbanProvider>
  );
};

export default CloudDevopsArchitectPage;
