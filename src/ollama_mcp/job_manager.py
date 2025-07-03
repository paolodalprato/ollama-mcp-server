"""
Job Manager - Ollama MCP Server v1.0.0
Manages background jobs for long-running operations

Design Principles:
- Type safety with full annotations
- Immutable configuration
- Comprehensive error handling
- Clean separation of concerns
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a background job"""
    job_id: str
    job_type: str
    status: JobStatus
    progress_percent: int = 0
    current_step: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def runtime_seconds(self) -> Optional[int]:
        """Calculate job runtime in seconds"""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or datetime.now()
        return int((end_time - self.started_at).total_seconds())


class JobManager:
    """
    Manager for background job execution
    
    Handles asynchronous operations that might take longer than
    MCP request timeouts, providing job tracking and status monitoring.
    """
    
    def __init__(self):
        """Initialize job manager"""
        self.jobs: Dict[str, Job] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        logger.debug("Initialized JobManager")
    
    def create_job(self, job_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job
        
        Args:
            job_type: Type of job (e.g., 'model_download')
            metadata: Optional job metadata
            
        Returns:
            Job ID string
        """
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            metadata=metadata or {}
        )
        
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id
    
    async def start_job(self, job_id: str, job_function: Callable[..., Awaitable[Dict[str, Any]]], 
                       *args, **kwargs) -> bool:
        """
        Start executing a job
        
        Args:
            job_id: Job ID to start
            job_function: Async function to execute
            *args, **kwargs: Arguments for job function
            
        Returns:
            True if job started successfully
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.PENDING:
            logger.error(f"Job {job_id} is not in pending status")
            return False
        
        try:
            # Create progress callback
            def progress_callback(percent: int, step: str):
                if job_id in self.jobs:
                    self.jobs[job_id].progress_percent = percent
                    self.jobs[job_id].current_step = step
            
            # Start the job task
            task = asyncio.create_task(
                self._execute_job(job_id, job_function, progress_callback, *args, **kwargs)
            )
            
            self.running_tasks[job_id] = task
            
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.current_step = "Job started"
            
            logger.info(f"Started job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            return False
    
    async def _execute_job(self, job_id: str, job_function: Callable, 
                          progress_callback: Callable, *args, **kwargs) -> None:
        """
        Execute job function and handle results
        
        Args:
            job_id: Job ID
            job_function: Function to execute
            progress_callback: Progress callback function
            *args, **kwargs: Function arguments
        """
        job = self.jobs[job_id]
        
        try:
            # Execute the job function
            result = await job_function(progress_callback, *args, **kwargs)
            
            # Update job with results
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now()
            job.progress_percent = 100
            job.current_step = "Completed"
            
            logger.info(f"Job {job_id} completed successfully")
            
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.current_step = "Cancelled"
            logger.info(f"Job {job_id} was cancelled")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            job.current_step = "Failed"
            logger.error(f"Job {job_id} failed: {e}")
            
        finally:
            # Clean up running task
            if job_id in self.running_tasks:
                del self.running_tasks[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Dict with job status or None if not found
        """
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status.value,
            "progress_percent": job.progress_percent,
            "current_step": job.current_step,
            "metadata": job.metadata,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "runtime_seconds": job.runtime_seconds
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.RUNNING:
            logger.warning(f"Job {job_id} is not running (status: {job.status.value})")
            return False
        
        # Cancel the running task
        if job_id in self.running_tasks:
            task = self.running_tasks[job_id]
            task.cancel()
            logger.info(f"Cancelled job {job_id}")
            return True
        
        # Job marked as running but no task found - mark as cancelled
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        job.current_step = "Cancelled"
        logger.info(f"Marked job {job_id} as cancelled")
        return True
    
    def list_jobs(self, job_type: Optional[str] = None, 
                  status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering
        
        Args:
            job_type: Filter by job type
            status: Filter by job status
            
        Returns:
            List of job status dictionaries
        """
        jobs = []
        
        for job in self.jobs.values():
            # Apply filters
            if job_type and job.job_type != job_type:
                continue
            if status and job.status != status:
                continue
            
            job_status = self.get_job_status(job.job_id)
            if job_status:
                jobs.append(job_status)
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x['created_at'], reverse=True)
        return jobs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get job manager statistics
        
        Returns:
            Dict with statistics
        """
        total_jobs = len(self.jobs)
        running_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING])
        completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
        cancelled_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.CANCELLED])
        
        return {
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "cancelled_jobs": cancelled_jobs,
            "active_tasks": len(self.running_tasks)
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed jobs
        
        Args:
            max_age_hours: Maximum age in hours to keep jobs
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            # Don't remove running jobs
            if job.status == JobStatus.RUNNING:
                continue
            
            # Check if job is old enough
            if job.created_at.timestamp() < cutoff_time:
                jobs_to_remove.append(job_id)
        
        # Remove old jobs
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
        return len(jobs_to_remove)


# Global job manager instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


# Export main classes
__all__ = [
    "JobManager",
    "Job",
    "JobStatus",
    "get_job_manager"
]
