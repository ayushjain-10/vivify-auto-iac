import { useState, useEffect, useCallback } from 'react';
import { GCPArchitecture } from '../types/gcp';
import { gcpApi } from '../services/gcpApi';
import { ApiError } from '../services/api';

export function useGCPArchitecture(
  project: string | null,
  credentials: any | null,
  hasGCPAccess: boolean,
  regions?: string[]
) {
  const [architecture, setArchitecture] = useState<GCPArchitecture | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!hasGCPAccess || !credentials || !project) {
      setLoading(false);
      setArchitecture(null);
      return;
    }
    
    setLoading(true);
    setError(null);
    console.log(`🔍 Requesting architecture discovery for project: ${project}`, regions);
    
    try {
      // Call real backend API
      const data = await gcpApi.discoverResources(credentials, project, regions);
      setArchitecture(data);
      console.log(`✅ Discovery complete: ${data.resources.length} resources found`);
    } catch (e) {
      console.error('❌ Discovery failed:', e);
      
      if (e instanceof ApiError) {
        setError(new Error(e.message));
      } else {
        setError(e instanceof Error ? e : new Error('An unknown error occurred during discovery.'));
      }
    } finally {
      setLoading(false);
    }
  }, [project, credentials, regions, hasGCPAccess]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const refresh = useCallback(() => {
    fetchData();
  }, [fetchData]);

  return { architecture, loading, error, refresh };
}