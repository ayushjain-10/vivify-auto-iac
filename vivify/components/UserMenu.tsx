import React, { useState, useRef } from 'react';
import { useGCPConnection } from '../context/GCPConnectionContext';
import { useGCPArchitectureStore } from '../context/GCPArchitectureStore';
import { useClickOutside } from '../hooks/useClickOutside';
import { UserCircleIcon } from './icons/UserCircleIcon';
import { KeyIcon } from './icons/KeyIcon';
import { XIcon } from './icons/XIcon';

const UserMenu: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { hasGCPAccess, clearGCPConnection, setIsModalOpen } = useGCPConnection();
  const { clearArchitecture } = useGCPArchitectureStore();
  const menuRef = useRef<HTMLDivElement>(null);

  useClickOutside(menuRef, () => setIsOpen(false));
  
  const handleManageCredentials = () => {
    setIsModalOpen(true);
    setIsOpen(false);
  };
  
  const handleDisconnect = () => {
    clearGCPConnection();
    clearArchitecture(); // Clear stored architecture data
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={menuRef}>
      <button onClick={() => setIsOpen(!isOpen)} className="rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-500">
        <UserCircleIcon className="w-8 h-8 text-gray-400 hover:text-white transition-colors" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 origin-top-right rounded-md bg-gray-700 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
          <div className="py-1" role="menu" aria-orientation="vertical" aria-labelledby="user-menu-button">
            <button
              onClick={handleManageCredentials}
              className="flex items-center w-full px-4 py-2 text-sm text-gray-200 hover:bg-gray-600 text-left"
              role="menuitem"
            >
              <KeyIcon className="w-5 h-5 mr-3" />
              <span>{hasGCPAccess ? 'Update' : 'Connect'} GCP Credentials</span>
            </button>
            {hasGCPAccess && (
              <button
                onClick={handleDisconnect}
                className="flex items-center w-full px-4 py-2 text-sm text-red-400 hover:bg-gray-600 text-left"
                role="menuitem"
              >
                <XIcon className="w-5 h-5 mr-3" />
                <span>Disconnect GCP</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default UserMenu;
