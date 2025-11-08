import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';

import { GCPArchitecture } from '../types/gcp';

const GCP_ACCESS_KEY = 'vibe_devops_gcp_access';
const GCP_CREDENTIALS_KEY = 'vibe_devops_gcp_credentials';
const GCP_PROJECT_KEY = 'vibe_devops_gcp_project';
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

interface GCPConnectionContextType {
  hasGCPAccess: boolean;
  credentials: any | null;
  projectId: string | null;
  setGCPConnection: (credentials: any, projectId: string) => void;
  clearGCPConnection: () => void;
  isModalOpen: boolean;
  setIsModalOpen: (open: boolean) => void;
  // Cache management
  architectureCache: GCPArchitecture | null;
  cacheTimestamp: number | null;
  setCachedArchitecture: (architecture: GCPArchitecture) => void;
  clearArchitectureCache: () => void;
  isCacheValid: () => boolean;
}

const GCPConnectionContext = createContext<GCPConnectionContextType | undefined>(undefined);

export const GCPConnectionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [hasGCPAccess, setHasGCPAccessState] = useState<boolean>(() => {
    try {
      return localStorage.getItem(GCP_ACCESS_KEY) === 'true';
    } catch {
      return false;
    }
  });
  
  const [credentials, setCredentialsState] = useState<any | null>(() => {
    try {
      const stored = localStorage.getItem(GCP_CREDENTIALS_KEY);
      return stored ? JSON.parse(atob(stored)) : null;
    } catch {
      return null;
    }
  });
  
  const [projectId, setProjectIdState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(GCP_PROJECT_KEY);
    } catch {
      return null;
    }
  });
  
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Architecture cache state
  const [architectureCache, setArchitectureCache] = useState<GCPArchitecture | null>(null);
  const [cacheTimestamp, setCacheTimestamp] = useState<number | null>(null);

  const setGCPConnection = (creds: any, project: string) => {
    try {
      // Store credentials (base64 encoded for basic obfuscation)
      const encoded = btoa(JSON.stringify(creds));
      localStorage.setItem(GCP_CREDENTIALS_KEY, encoded);
      localStorage.setItem(GCP_PROJECT_KEY, project);
      localStorage.setItem(GCP_ACCESS_KEY, 'true');
      
      setCredentialsState(creds);
      setProjectIdState(project);
      setHasGCPAccessState(true);
    } catch (error) {
      console.error("Failed to save GCP connection:", error);
    }
  };

  const clearGCPConnection = () => {
    try {
      localStorage.removeItem(GCP_CREDENTIALS_KEY);
      localStorage.removeItem(GCP_PROJECT_KEY);
      localStorage.removeItem(GCP_ACCESS_KEY);
      
      setCredentialsState(null);
      setProjectIdState(null);
      setHasGCPAccessState(false);
    } catch (error) {
      console.error("Failed to clear GCP connection:", error);
    }
  };
  
  const value = useMemo(() => ({
    hasGCPAccess,
    credentials,
    projectId,
    setGCPConnection,
    clearGCPConnection,
    isModalOpen,
    setIsModalOpen,
  }), [hasGCPAccess, credentials, projectId, isModalOpen]);

  return (
    <GCPConnectionContext.Provider value={value}>
      {children}
    </GCPConnectionContext.Provider>
  );
};

export const useGCPConnection = (): GCPConnectionContextType => {
  const context = useContext(GCPConnectionContext);
  if (!context) {
    throw new Error('useGCPConnection must be used within a GCPConnectionProvider');
  }
  return context;
};
