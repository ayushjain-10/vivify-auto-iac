"""
Authentication and credential validation utilities
"""

import json
import tempfile
from typing import Dict, Any, Tuple
from google.oauth2 import service_account
from google.auth.exceptions import GoogleAuthError
from google.cloud import resourcemanager_v3


def validate_service_account_credentials(credentials: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Validate GCP service account credentials
    
    Args:
        credentials: Service account JSON as dictionary
        
    Returns:
        Tuple of (is_valid, project_id, error_message)
    """
    try:
        # Check required fields
        required_fields = ["type", "project_id", "private_key", "client_email"]
        missing_fields = [field for field in required_fields if field not in credentials]
        
        if missing_fields:
            return False, "", f"Missing required fields: {', '.join(missing_fields)}"
        
        # Verify it's a service account
        if credentials.get("type") != "service_account":
            return False, "", "Credentials must be for a service account"
        
        # Extract project ID
        project_id = credentials.get("project_id", "")
        
        # Create credentials object
        creds = service_account.Credentials.from_service_account_info(credentials)
        
        # Try to use the credentials to list projects (minimal permission test)
        # This verifies the credentials are valid and have at least viewer access
        try:
            client = resourcemanager_v3.ProjectsClient(credentials=creds)
            # Try to get the project
            project_name = f"projects/{project_id}"
            project = client.get_project(name=project_name)
            
            return True, project_id, ""
            
        except GoogleAuthError as e:
            return False, project_id, f"Authentication failed: {str(e)}"
        except Exception as e:
            # Credentials might be valid but lack permissions
            # We'll allow this and let discovery handle permission errors
            return True, project_id, ""
    
    except Exception as e:
        return False, "", f"Invalid credentials format: {str(e)}"


def credentials_to_file(credentials: Dict[str, Any]) -> str:
    """
    Write credentials to a temporary file and return the path
    Useful for GCP client libraries that require a file path
    
    Args:
        credentials: Service account JSON as dictionary
        
    Returns:
        Path to temporary credentials file
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(credentials, temp_file)
    temp_file.close()
    return temp_file.name


def get_credentials_object(credentials: Dict[str, Any]) -> service_account.Credentials:
    """
    Convert credentials dictionary to Google credentials object
    
    Args:
        credentials: Service account JSON as dictionary
        
    Returns:
        Google service account credentials object
    """
    return service_account.Credentials.from_service_account_info(credentials)
