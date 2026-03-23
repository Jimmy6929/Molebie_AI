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
            if not response.is_success:
                print(
                    f"[database] {method} {endpoint} → {response.status_code}: "
                    f"{response.text[:500]}"
                )
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
        """List user's chat sessions, pinned first then newest."""
        result = self._request(
            "GET",
            f"chat_sessions?user_id=eq.{user_id}&is_archived=eq.false&order=is_pinned.desc,updated_at.desc&limit={limit}",
            user_token=user_token,
        )
        return result or []

    def pin_session(self, session_id: str, user_id: str, is_pinned: bool, user_token: Optional[str] = None) -> bool:
        """Pin or unpin a session."""
        result = self._request(
            "PATCH",
            f"chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
            user_token=user_token,
            json={"is_pinned": is_pinned},
        )
        return bool(result)

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
        reasoning_content: Optional[str] = None,
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
        if reasoning_content:
            data["reasoning_content"] = reasoning_content
            
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
    
    # ==================== Message Images ====================

    def create_message_image(
        self,
        message_id: str,
        user_id: str,
        storage_path: str,
        filename: Optional[str] = None,
        mime_type: str = "image/jpeg",
        file_size: int = 0,
        user_token: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store image metadata for a chat message."""
        data = {
            "message_id": message_id,
            "user_id": user_id,
            "storage_path": storage_path,
            "mime_type": mime_type,
            "file_size": file_size,
        }
        if filename:
            data["filename"] = filename
        result = self._request("POST", "message_images", user_token=user_token, json=data)
        return result[0] if result else None

    def get_message_images(
        self,
        message_ids: List[str],
        user_id: str,
        user_token: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch image metadata for a list of message IDs.

        Returns a dict keyed by message_id with image info (storage_path, filename, mime_type, id).
        """
        if not message_ids:
            return {}
        result = self._request(
            "GET",
            f"message_images?message_id=in.({','.join(message_ids)})&user_id=eq.{user_id}"
            f"&select=id,message_id,storage_path,filename,mime_type,file_size",
            user_token=user_token,
        )
        images = {}
        for row in (result or []):
            images[row["message_id"]] = row
        return images

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
