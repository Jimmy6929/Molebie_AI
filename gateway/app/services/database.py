"""
Database service for Supabase interactions via direct REST API.
"""

from typing import Optional, List, Dict, Any
from functools import lru_cache
import httpx

from app.config import Settings, get_settings


class DatabaseService:
    """Service for database operations via Supabase REST API."""
    
    def __init__(self, settings: Settings):
        self.base_url = f"{settings.supabase_url}/rest/v1"
        self.anon_key = settings.supabase_anon_key
    
    def _build_headers(self, user_token: Optional[str] = None) -> dict:
        """Build request headers, using the user's JWT for RLS auth."""
        return {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {user_token}" if user_token else f"Bearer {self.anon_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    
    def _request(self, method: str, endpoint: str, user_token: Optional[str] = None, **kwargs) -> Any:
        """Make a synchronous HTTP request to Supabase REST API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {**self._build_headers(user_token), **kwargs.pop("headers", {})}
        
        with httpx.Client() as client:
            response = client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json() if response.text else None
    
    # ==================== Sessions ====================
    
    def create_session(self, user_id: str, title: str = "New Chat", user_token: Optional[str] = None) -> Dict[str, Any]:
        """Create a new chat session."""
        result = self._request("POST", "chat_sessions", user_token=user_token, json={
            "user_id": user_id,
            "title": title,
        })
        return result[0] if result else None
    
    def get_session(self, session_id: str, user_id: str, user_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a session by ID (only if owned by user)."""
        result = self._request(
            "GET",
            f"chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}&select=*",
            user_token=user_token,
        )
        return result[0] if result else None
    
    def list_sessions(self, user_id: str, limit: int = 50, user_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """List user's chat sessions, newest first."""
        result = self._request(
            "GET",
            f"chat_sessions?user_id=eq.{user_id}&is_archived=eq.false&order=updated_at.desc&limit={limit}",
            user_token=user_token,
        )
        return result or []
    
    def update_session_title(self, session_id: str, user_id: str, title: str, user_token: Optional[str] = None) -> bool:
        """Update session title."""
        result = self._request(
            "PATCH",
            f"chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
            user_token=user_token,
            json={"title": title},
        )
        return bool(result)
    
    def delete_session(self, session_id: str, user_id: str, user_token: Optional[str] = None) -> bool:
        """Delete a session (cascades to messages)."""
        self._request(
            "DELETE",
            f"chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
            user_token=user_token,
        )
        return True
    
    # ==================== Messages ====================
    
    def create_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        mode_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        user_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new chat message."""
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
        }
        if mode_used:
            data["mode_used"] = mode_used
        if tokens_used:
            data["tokens_used"] = tokens_used
            
        result = self._request("POST", "chat_messages", user_token=user_token, json=data)
        return result[0] if result else None
    
    def get_session_messages(
        self,
        session_id: str,
        user_id: str,
        limit: int = 100,
        user_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get messages for a session, oldest first."""
        result = self._request(
            "GET",
            f"chat_messages?session_id=eq.{session_id}&user_id=eq.{user_id}&order=created_at.asc&limit={limit}",
            user_token=user_token,
        )
        return result or []
    
    # ==================== User Profile ====================
    
    def get_or_create_profile(self, user_id: str, email: Optional[str] = None, user_token: Optional[str] = None) -> Dict[str, Any]:
        """Get user profile, creating if needed."""
        result = self._request("GET", f"profiles?id=eq.{user_id}&select=*", user_token=user_token)
        if result:
            return result[0]
        
        if email:
            result = self._request("POST", "profiles", user_token=user_token, json={
                "id": user_id,
                "email": email,
            })
            return result[0] if result else None
        
        return None


@lru_cache
def get_database_service() -> DatabaseService:
    """Get cached database service instance."""
    return DatabaseService(get_settings())
