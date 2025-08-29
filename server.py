from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from supabase import create_client, Client
import json
import os
import logging
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
import httpx
import hmac
import hashlib

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase client
supabase: Optional[Client] = None

# API base URLs
CHAT_API_BASE_URL = os.getenv("CHAT_API_BASE_URL", "http://82.112.227.182:8080")
SEARCH_API_BASE_URL = os.getenv("SEARCH_API_BASE_URL")

# Pydantic models
class InitializeProjectRequest(BaseModel):
    project_id: str
    username: str
    project_name: str
    chat_api_key: str
    chat_templates_config: Optional[Dict[str, Any]] = None
    search_api_key: str
    search_templates_config: Optional[Dict[str, Any]] = None

class ReadTemplateConfigRequest(BaseModel):
    product_type: Literal['chat', 'search']
    project_id: str
    template_name: str

class ModifyTemplateConfigRequest(BaseModel):
    product_type: Literal['chat', 'search']
    api_key: str
    template_name: str
    updated_config: Dict[str, Any]

class DeleteProjectRequest(BaseModel):
    api_key: str

class TemplateConfigResponse(BaseModel):
    template_config: Dict[str, Any]

class SuccessResponse(BaseModel):
    message: str
    success: bool = True

# Updated request models for proxy routes with HMAC signature
class ProxyNewUserMessageRequest(BaseModel):
    project_id: str
    user_message: str
    consumer_id: str
    hmac_signature: str
    chat_id: Optional[str] = None
    test: Optional[bool] = False

class ProxyListConsumerChatsRequest(BaseModel):
    project_id: str
    consumer_id: str
    hmac_signature: str
    test: Optional[bool] = False

class ProxyReadConversationRequest(BaseModel):
    project_id: str
    chat_id: str
    test: Optional[bool] = False

# New request models for search proxy routes
class ProxyAutocompleteRequest(BaseModel):
    project_id: str
    consumer_id: str
    hmac_signature: str
    query: str
    test: Optional[bool] = False
    max_results: Optional[int] = 10

class ProxySearchRequest(BaseModel):
    project_id: str
    consumer_id: str
    hmac_signature: str
    query: str
    search_type: str = "keyword"  # "keyword" or "hybrid"
    test: Optional[bool] = False
    limit: Optional[int] = None
    offset: Optional[int] = None

# Supabase client function
def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not available")
    return supabase

# Helper function to get API key by project_id from chat_api_keys table
async def get_api_key_by_project_id(project_id: str) -> str:
    """Get API key from chat_api_keys table using project_id"""
    client = get_supabase_client()
    
    try:
        result = client.table("chat_api_keys").select("api_key").eq("project_id", project_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No API key found for project_id: {project_id}"
            )
        
        return result.data[0]["api_key"]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving API key for project_id {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving API key: {str(e)}")

# Helper function to get search API key by project_id from search_api_keys table
async def get_search_api_key_by_project_id(project_id: str) -> str:
    """Get search API key from search_api_keys table using project_id"""
    client = get_supabase_client()
    
    try:
        result = client.table("search_api_keys").select("api_key").eq("project_id", project_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No search API key found for project_id: {project_id}"
            )
        
        return result.data[0]["api_key"]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving search API key for project_id {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving search API key: {str(e)}")

# Helper function to validate HMAC signature
def validate_hmac_signature(api_key: str, consumer_id: str, provided_signature: str) -> bool:
    """Validate HMAC signature using API key and consumer ID"""
    try:
        expected_signature = hmac.new(
            api_key.encode('utf-8'),
            consumer_id.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, provided_signature)
    except Exception as e:
        logger.error(f"Error validating HMAC signature: {e}")
        return False

# Lifespan context manager for Supabase client initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase
    
    # Startup
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Supabase client cleanup completed")

# FastAPI app
app = FastAPI(
    title="Templates Config Service",
    description="API for managing template configurations in Supabase",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/initialize-new-project", response_model=SuccessResponse)
async def initialize_new_project(request: InitializeProjectRequest):
    """Add a new project configuration to the database"""
    client = get_supabase_client()
    
    try:
        # Check if project_id already exists
        existing_project = client.table("templates_config").select("id").eq("project_id", request.project_id).execute()
        
        if existing_project.data:
            raise HTTPException(
                status_code=400, 
                detail=f"Project with ID {request.project_id} already exists"
            )
        
        # Prepare data for insertion
        insert_data = {
            "project_id": request.project_id,
            "username": request.username,
            "project_name": request.project_name,
            "chat_api_key": request.chat_api_key,
            "search_api_key": request.search_api_key
        }
        
        # Add optional JSON configs
        if request.chat_templates_config is not None:
            insert_data["chat_templates_config"] = request.chat_templates_config
        
        if request.search_templates_config is not None:
            insert_data["search_templates_config"] = request.search_templates_config
        
        # Insert new project
        result = client.table("templates_config").insert(insert_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create project")
        
        logger.info(f"Successfully created project: {request.project_id}")
        return SuccessResponse(message="Project initialized successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing project: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/read-template-config", response_model=TemplateConfigResponse)
async def read_template_config(request: ReadTemplateConfigRequest):
    """Read a specific template configuration"""
    client = get_supabase_client()
    
    try:
        # Determine which API key field and config field to use
        project_id_field = 'project_id'
        if request.product_type == 'chat':
            config_field = 'chat_templates_config'
        else:  # search
            config_field = 'search_templates_config'
        
        # Query the database
        result = client.table("templates_config").select(config_field).eq(project_id_field, request.project_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail=f"No project found for the provided project_id"
            )
        
        templates_config = result.data[0][config_field]
        
        if not templates_config:
            raise HTTPException(
                status_code=404, 
                detail=f"No {request.product_type} templates configuration found"
            )
        
        # Check if template_name exists in the configuration
        if request.template_name not in templates_config:
            raise HTTPException(
                status_code=404, 
                detail=f"Template '{request.template_name}' not found in {request.product_type} configuration"
            )
        
        template_config = templates_config[request.template_name]
        
        logger.info(f"Successfully retrieved template config for: {request.template_name}")
        return TemplateConfigResponse(template_config=template_config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading template config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/modify-template-config", response_model=SuccessResponse)
async def modify_template_config(request: ModifyTemplateConfigRequest):
    """Modify a specific template configuration"""
    client = get_supabase_client()
    
    try:
        # Determine which API key field and config field to use
        if request.product_type == 'chat':
            api_key_field = 'chat_api_key'
            config_field = 'chat_templates_config'
        else:  # search
            api_key_field = 'search_api_key'
            config_field = 'search_templates_config'
        
        # First, get the current configuration
        result = client.table("templates_config").select(config_field).eq(api_key_field, request.api_key).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail=f"No project found with the provided {request.product_type} API key"
            )
        
        current_config = result.data[0][config_field] or {}
        
        # Update the specific template
        current_config[request.template_name] = request.updated_config
        
        # Update the database
        update_result = client.table("templates_config").update({
            config_field: current_config
        }).eq(api_key_field, request.api_key).execute()
        
        if not update_result.data:
            raise HTTPException(status_code=404, detail="Project not found or update failed")
        
        logger.info(f"Successfully modified template config for: {request.template_name}")
        return SuccessResponse(message="Template configuration updated successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying template config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/delete-project", response_model=SuccessResponse)
async def delete_project(request: DeleteProjectRequest):
    """Delete a project by API key"""
    client = get_supabase_client()
    
    try:
        # First check if project exists with either chat_api_key or search_api_key
        chat_result = client.table("templates_config").select("id").eq("chat_api_key", request.api_key).execute()
        search_result = client.table("templates_config").select("id").eq("search_api_key", request.api_key).execute()
        
        if not chat_result.data and not search_result.data:
            raise HTTPException(
                status_code=404, 
                detail="No project found with the provided API key"
            )
        
        # Delete by chat_api_key
        if chat_result.data:
            delete_result = client.table("templates_config").delete().eq("chat_api_key", request.api_key).execute()
        # Delete by search_api_key
        else:
            delete_result = client.table("templates_config").delete().eq("search_api_key", request.api_key).execute()
        
        if not delete_result.data:
            raise HTTPException(status_code=500, detail="Failed to delete project")
        
        logger.info(f"Successfully deleted project with API key: {request.api_key[:10]}...")
        return SuccessResponse(message="Project deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Updated proxy routes with HMAC validation
@app.post("/proxy-new-user-message")
async def proxy_new_user_message(request: ProxyNewUserMessageRequest):
    """Proxy route for /newUserMessage - gets API key by project_id and forwards request with HMAC validation"""
    try:
        # Get API key using project_id
        api_key = await get_api_key_by_project_id(request.project_id)
        
        # Validate HMAC signature if not 'anon'
        if request.hmac_signature != 'anon':
            if not validate_hmac_signature(api_key, request.consumer_id, request.hmac_signature):
                raise HTTPException(
                    status_code=401,
                    detail="HMAC signature validation failed. Authentication error."
                )
        
        # Prepare request payload for the chat service
        chat_request_payload = {
            "user_message": request.user_message,
            "chat_id": request.chat_id,
            "consumer_id": request.consumer_id,
            "test": request.test
        }
        
        # Make request to chat service
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            response = await client.post(
                f"{CHAT_API_BASE_URL}/newUserMessage",
                json=chat_request_payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Chat service returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Chat service error: {response.text}"
                )
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in proxy_new_user_message: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/proxy-list-consumer-chats")
async def proxy_list_consumer_chats(
    project_id: str,
    consumer_id: str,
    hmac_signature: str,
    test: bool = False
):
    """Proxy route for /listAllProjectConsumerChats - gets API key by project_id and forwards request with HMAC validation"""
    try:
        # Get API key using project_id
        api_key = await get_api_key_by_project_id(project_id)
        
        # Validate HMAC signature if not 'anon'
        if hmac_signature != 'anon':
            if not validate_hmac_signature(api_key, consumer_id, hmac_signature):
                raise HTTPException(
                    status_code=401,
                    detail="HMAC signature validation failed. Authentication error."
                )
        
        # Prepare query parameters
        params = {
            "consumer_id": consumer_id,
            "test": test
        }
        
        # Make request to chat service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{CHAT_API_BASE_URL}/listAllProjectConsumerChats",
                params=params,
                headers={
                    "Authorization": f"Bearer {api_key}"
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Chat service returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Chat service error: {response.text}"
                )
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in proxy_list_consumer_chats: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/proxy-read-conversation")
async def proxy_read_conversation(
    project_id: str,
    chat_id: str,
    test: bool = False
):
    """Proxy route for /readConversation - gets API key by project_id and forwards request"""
    try:
        # Get API key using project_id
        api_key = await get_api_key_by_project_id(project_id)
        
        # Prepare query parameters
        params = {
            "chat_id": chat_id,
            "test": test
        }
        
        # Make request to chat service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{CHAT_API_BASE_URL}/readConversation",
                params=params,
                headers={
                    "Authorization": f"Bearer {api_key}"
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Chat service returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Chat service error: {response.text}"
                )
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in proxy_read_conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# New search proxy routes
@app.post("/proxy-autocomplete")
async def proxy_autocomplete(request: ProxyAutocompleteRequest):
    """Proxy route for /autocomplete - gets search API key by project_id and forwards request with HMAC validation"""
    try:
        if not SEARCH_API_BASE_URL:
            raise HTTPException(
                status_code=500,
                detail="SEARCH_API_BASE_URL environment variable is not configured"
            )
        
        # Get search API key using project_id
        api_key = await get_search_api_key_by_project_id(request.project_id)
        
        # Validate HMAC signature if not 'anon'
        if request.hmac_signature != 'anon':
            if not validate_hmac_signature(api_key, request.consumer_id, request.hmac_signature):
                raise HTTPException(
                    status_code=401,
                    detail="HMAC signature validation failed. Authentication error."
                )
        
        # Prepare request payload for the search service
        autocomplete_request_payload = {
            "consumer_id": request.consumer_id,
            "autocomplete_term": request.query,
            "limit": request.max_results,
            "test": request.test
        }
        
        # Make request to search service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SEARCH_API_BASE_URL}/autocomplete",
                json=autocomplete_request_payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Search service returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Search service error: {response.text}"
                )
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in proxy_autocomplete: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.post("/proxy-search")
async def proxy_search(request: ProxySearchRequest):
    """Proxy route for /search - gets search API key by project_id and forwards request with HMAC validation"""
    try:
        if not SEARCH_API_BASE_URL:
            raise HTTPException(
                status_code=500,
                detail="SEARCH_API_BASE_URL environment variable is not configured"
            )
        
        # Get search API key using project_id
        api_key = await get_search_api_key_by_project_id(request.project_id)
        
        # Validate HMAC signature if not 'anon'
        if request.hmac_signature != 'anon':
            if not validate_hmac_signature(api_key, request.consumer_id, request.hmac_signature):
                raise HTTPException(
                    status_code=401,
                    detail="HMAC signature validation failed. Authentication error."
                )
        
        # Prepare request payload for the search service
        search_request_payload = {
            "consumer_id": request.consumer_id,
            "search_term": request.query,
            "search_type": request.search_type,
            "semantic_ratio": 0.7,
            "limit": request.limit, 
            "offset": request.offset,
            "test": request.test
        }
        
        # Make request to search service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SEARCH_API_BASE_URL}/search",
                json=search_request_payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Search service returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Search service error: {response.text}"
                )
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in proxy_search: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "templates-config-service"}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8500,
        log_level="info",
        reload=False
    )