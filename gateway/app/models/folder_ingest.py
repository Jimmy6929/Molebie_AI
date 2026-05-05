"""
Pydantic models for the folder-ingest endpoints.
"""

from pydantic import BaseModel, Field


class FolderManifestEntry(BaseModel):
    relative_path: str = Field(..., description="Path within the dropped folder, with /-separators.")
    size: int = Field(..., ge=0)
    content_type: str | None = None


class StartFolderJobRequest(BaseModel):
    root_label: str = Field(..., description="Top-level folder name shown to the user.")
    files: list[FolderManifestEntry]


class AcceptedFile(BaseModel):
    file_id: str
    relative_path: str
    size: int


class RejectedFile(BaseModel):
    relative_path: str
    reason: str


class StartFolderJobResponse(BaseModel):
    job_id: str
    accepted_files: list[AcceptedFile]
    rejected_files: list[RejectedFile]
    total_accepted_bytes: int


class UploadFolderBatchResponse(BaseModel):
    accepted: list[str] = Field(default_factory=list, description="file_ids stored")
    rejected: list[RejectedFile] = Field(default_factory=list)


class FolderJobSnapshot(BaseModel):
    job_id: str
    status: str
    root_label: str
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    total_bytes: int
    processed_bytes: int
    started_at: str | None = None
    finished_at: str | None = None
    last_event_id: int = 0


class CancelFolderJobResponse(BaseModel):
    job_id: str
    status: str
