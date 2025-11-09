/**
 * GCP Architecture Store
 * Global state management for GCP architecture data
 * 
 * Key Features:
 * - NO automatic fetching
 * - Manual refresh only (user clicks button)
 * - Persists across tab switches
 * - Accessible from anywhere in the app
 */

import { create } from 'zustand';
import { GCPArchitecture } from '../types/gcp';
import { gcpApi } from '../services/gcpApi';

interface GCPArchitectureState {
  // State
  architecture: GCPArchitecture | null;
  loading: boolean;
  error: Error | null;
  lastFetch: Date | null;

  // Actions
  fetchArchitecture: (credentials: any, project: string, regions?: string[]) => Promise<void>;
  clearArchitecture: () => void;
}

export const useGCPArchitectureStore = create<GCPArchitectureState>((set) => ({
  // Initial state
  architecture: null,
  loading: false,
  error: null,
  lastFetch: null,

  // Manual fetch only - NO automatic fetching
  fetchArchitecture: async (credentials: any, project: string, regions?: string[]) => {
    set({ loading: true, error: null });
    
    try {
      console.log('🔍 Manual fetch triggered for project:', project);
      
      const data = await gcpApi.discoverResources(credentials, project, regions);
      
      set({
        architecture: data,
        loading: false,
        error: null,
        lastFetch: new Date(),
      });
      
      console.log('✅ Architecture data stored:', data.resources.length, 'resources');
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to fetch architecture');
      console.error('❌ Fetch failed:', error.message);
      
      set({
        loading: false,
        error,
      });
    }
  },

  // Clear architecture data
  clearArchitecture: () => {
    console.log('🗑️ Clearing architecture data');
    set({
      architecture: null,
      loading: false,
      error: null,
      lastFetch: null,
    });
  },
}));
