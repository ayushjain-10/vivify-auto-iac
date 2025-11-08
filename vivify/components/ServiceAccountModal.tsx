import React, { useState, useCallback } from 'react';
import { useGCPConnection } from '../context/GCPConnectionContext';
import { gcpApi } from '../services/gcpApi';
import { ApiError } from '../services/api';
import { XIcon } from './icons/XIcon';
import { UploadCloudIcon } from './icons/UploadCloudIcon';
import { FileJsonIcon } from './icons/FileJsonIcon';
import { CheckCircleIcon } from './icons/CheckCircleIcon';
import { XCircleIcon } from './icons/XCircleIcon';
import { SpinnerIcon } from './icons/SpinnerIcon';

interface ServiceAccountModalProps {
  onClose: () => void;
}

type FileValidationStatus = 'idle' | 'validating' | 'valid' | 'invalid';
type ConnectionStatus = 'idle' | 'connecting' | 'testing' | 'success' | 'failed';


const ServiceAccountModal: React.FC<ServiceAccountModalProps> = ({ onClose }) => {
  const { setGCPConnection } = useGCPConnection();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const [fileValidationStatus, setFileValidationStatus] = useState<FileValidationStatus>('idle');
  const [fileValidationError, setFileValidationError] = useState<string | null>(null);

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('idle');
  const [connectionMessage, setConnectionMessage] = useState<string | null>(null);


  const resetConnectionState = () => {
    setConnectionStatus('idle');
    setConnectionMessage(null);
  }

  const validateFile = useCallback((uploadedFile: File | null) => {
    resetConnectionState();
    if (!uploadedFile) {
        setFileValidationStatus('idle');
        return;
    }

    setFileValidationStatus('validating');
    setFileValidationError(null);

    if (!uploadedFile.name.endsWith('.json')) {
        setFileValidationError('File must be a .json file.');
        setFileValidationStatus('invalid');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const content = JSON.parse(e.target?.result as string);
            if (content.project_id && content.private_key && content.client_email) {
                setFileValidationStatus('valid');
            } else {
                setFileValidationError('JSON is missing required fields (project_id, private_key, client_email).');
                setFileValidationStatus('invalid');
            }
        } catch (error) {
            setFileValidationError('File is not valid JSON.');
            setFileValidationStatus('invalid');
        }
    };
    reader.readAsText(uploadedFile);
  }, []);

  const handleFileChange = (files: FileList | null) => {
    if (files && files.length > 0) {
      const newFile = files[0];
      setFile(newFile);
      validateFile(newFile);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileChange(e.dataTransfer.files);
  };
  
  const handleConnect = async () => {
    if (!file) return;

    setConnectionStatus('connecting');
    setConnectionMessage('Preparing to connect...');
    
    try {
        const fileContent = await file.text();
        const credentials = JSON.parse(fileContent);

        setConnectionStatus('testing');
        setConnectionMessage('Validating credentials with GCP...');
        
        // Call real backend API
        const result = await gcpApi.validateCredentials(credentials);

        if (!result.valid) {
            throw new Error(result.error || 'Invalid credentials');
        }
        
        if (!result.projectId) {
            throw new Error('Could not determine project ID from credentials');
        }
        
        setConnectionStatus('success');
        setConnectionMessage(`Successfully connected to project: ${result.projectId}`);
        
        // Store credentials and project ID
        setGCPConnection(credentials, result.projectId);
        
        setTimeout(onClose, 2000); // Close modal after success message

    } catch (error) {
        setConnectionStatus('failed');
        
        if (error instanceof ApiError) {
            setConnectionMessage(error.message);
        } else {
            setConnectionMessage(error instanceof Error ? error.message : 'An unknown error occurred.');
        }
    }
  };

  const dragProps = {
    onDragEnter: (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setIsDragging(true); },
    onDragLeave: (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setIsDragging(false); },
    onDragOver: (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); },
    onDrop: handleDrop,
  };

  const renderFooterContent = () => {
    switch(connectionStatus) {
      case 'connecting':
      case 'testing':
        return (
          <div className="flex items-center justify-center space-x-2 text-gray-300">
            <SpinnerIcon className="w-5 h-5" />
            <span>{connectionMessage}</span>
          </div>
        );
      case 'success':
        return (
          <div className="flex items-center justify-center space-x-2 text-green-400">
            <CheckCircleIcon className="w-5 h-5" />
            <span>{connectionMessage}</span>
          </div>
        );
      case 'failed':
        return (
          <div className="text-center">
             <div className="flex items-center justify-center space-x-2 text-red-400">
                <XCircleIcon className="w-5 h-5" />
                <p>{connectionMessage}</p>
             </div>
            <button
              onClick={() => handleConnect()}
              className="mt-2 text-sm text-blue-400 hover:underline"
            >
              Retry
            </button>
          </div>
        );
      case 'idle':
      default:
        return (
            <button
                onClick={handleConnect}
                disabled={fileValidationStatus !== 'valid'}
                className="w-full px-4 py-2 bg-blue-600 text-white font-semibold rounded-md transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-blue-500"
            >
                Connect to GCP Project
            </button>
        );
    }
  }


  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex justify-center items-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        <header className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Connect to Google Cloud</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors disabled:opacity-50" disabled={connectionStatus === 'connecting' || connectionStatus === 'testing' || connectionStatus === 'success'}>
            <XIcon className="w-6 h-6" />
          </button>
        </header>

        <main className="flex-1 p-6 grid md:grid-cols-2 gap-6 overflow-y-auto">
          {/* Instructions */}
          <div className="space-y-4 text-sm text-gray-300">
            <h3 className="text-lg font-semibold text-white">Instructions</h3>
            <p>To connect your GCP project, create a Service Account with <strong className="text-yellow-400">read-only</strong> permissions.</p>
            <ol className="list-decimal list-inside space-y-2 text-gray-400">
              <li>Go to the <a href="https://console.cloud.google.com/iam-admin/serviceaccounts" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">Service Accounts</a> page in your GCP Console.</li>
              <li>Click <strong className="text-gray-200">CREATE SERVICE ACCOUNT</strong>.</li>
              <li>Give it a name (e.g., `vibe-devops-viewer`).</li>
              <li>Grant it the <strong className="text-gray-200">Viewer</strong> role. This provides the necessary read-only access.</li>
              <li>Click <strong className="text-gray-200">Done</strong>. Then, find your new service account, click the three dots under "Actions", and select <strong className="text-gray-200">Manage keys</strong>.</li>
              <li>Click <strong className="text-gray-200">ADD KEY</strong> &rarr; <strong className="text-gray-200">Create new key</strong>, choose <strong className="text-gray-200">JSON</strong>, and click <strong className="text-gray-200">CREATE</strong>.</li>
              <li>A JSON file will download. Upload that file here.</li>
            </ol>
          </div>

          {/* Uploader */}
          <div className="flex flex-col space-y-4">
            <h3 className="text-lg font-semibold text-white">Upload Key File</h3>
            <div 
              {...dragProps}
              className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-lg transition-colors ${isDragging ? 'border-blue-500 bg-gray-700/50' : 'border-gray-600 hover:border-gray-500'}`}
            >
              <UploadCloudIcon className="w-10 h-10 text-gray-500 mb-2" />
              <p className="text-sm text-gray-400">
                <label htmlFor="file-upload" className="font-semibold text-blue-400 cursor-pointer hover:underline">
                  Click to upload
                </label>
                {' '}or drag and drop
              </p>
              <p className="text-xs text-gray-500">Service Account JSON key file</p>
              <input id="file-upload" type="file" className="hidden" accept=".json" onChange={(e) => handleFileChange(e.target.files)} />
            </div>

            {file && (
              <div className="flex items-center p-3 bg-gray-700 rounded-md">
                <FileJsonIcon className="w-6 h-6 text-gray-400 flex-shrink-0" />
                <div className="ml-3 text-sm flex-1 overflow-hidden">
                  <p className="font-medium text-white truncate">{file.name}</p>
                  <p className="text-gray-400">{(file.size / 1024).toFixed(2)} KB</p>
                </div>
                {fileValidationStatus === 'valid' && <CheckCircleIcon className="w-6 h-6 text-green-500 ml-3" />}
                {fileValidationStatus === 'invalid' && <XCircleIcon className="w-6 h-6 text-red-500 ml-3" />}
              </div>
            )}
            {fileValidationError && <p className="text-xs text-red-400 text-center">{fileValidationError}</p>}
          </div>
        </main>
        
        <footer className="p-4 bg-gray-900/50 border-t border-gray-700 h-16 flex items-center justify-center">
            {renderFooterContent()}
        </footer>
      </div>
    </div>
  );
};

export default ServiceAccountModal;