"""
Ollama Server Control Module
Gestisce start/stop/controllo del server Ollama locale per MCP v0.9
Basato su Bridge v1.0 funzionante
"""

import asyncio
import logging
import os
import platform
import psutil
import subprocess
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class OllamaProcessInfo:
    """Informazioni sul processo Ollama"""
    pid: Optional[int]
    status: str  # "running", "stopped", "unknown"
    port: int
    uptime_seconds: Optional[int]
    memory_mb: Optional[float]
    cpu_percent: Optional[float]


class OllamaServerController:
    """Controller per gestire il server Ollama"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.port = 11434
        self.logger = logging.getLogger(__name__)
        
    async def get_server_status(self) -> OllamaProcessInfo:
        """Ottieni status dettagliato del server Ollama"""
        try:
            # Cerca processo Ollama
            ollama_pid = None
            ollama_process = None
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                        # Verifica che sia il server (non client commands)
                        cmdline = proc.info.get('cmdline', [])
                        if any('serve' in str(cmd).lower() for cmd in cmdline) or \
                           any('11434' in str(cmd) for cmd in cmdline):
                            ollama_pid = proc.info['pid']
                            ollama_process = proc
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if ollama_process:
                try:
                    # Ottieni info dettagliate processo
                    memory_info = ollama_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    cpu_percent = ollama_process.cpu_percent()
                    create_time = ollama_process.create_time()
                    uptime = int(time.time() - create_time)
                    
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.port,
                        uptime_seconds=uptime,
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.warning(f"Could not get process details: {e}")
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.port,
                        uptime_seconds=None,
                        memory_mb=None,
                        cpu_percent=None
                    )
            else:
                return OllamaProcessInfo(
                    pid=None,
                    status="stopped",
                    port=self.port,
                    uptime_seconds=None,
                    memory_mb=None,
                    cpu_percent=None
                )
                
        except Exception as e:
            self.logger.error(f"Error getting Ollama status: {e}")
            return OllamaProcessInfo(
                pid=None,
                status="unknown",
                port=self.port,
                uptime_seconds=None,
                memory_mb=None,
                cpu_percent=None
            )
    
    async def start_server(self) -> Dict[str, Any]:
        """Avvia il server Ollama"""
        try:
            # Verifica se già in esecuzione
            status = await self.get_server_status()
            if status.status == "running":
                return {
                    "success": True,
                    "message": "Ollama server già in esecuzione",
                    "pid": status.pid,
                    "already_running": True
                }
            
            # Determina comando di avvio basato su OS
            if platform.system() == "Windows":
                # Su Windows, ollama serve si avvia in background
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Su Linux/macOS
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Attendi che il server si avvii (max 10 secondi)
            for i in range(10):
                await asyncio.sleep(1)
                status = await self.get_server_status()
                if status.status == "running":
                    return {
                        "success": True,
                        "message": "Ollama server avviato con successo",
                        "pid": status.pid,
                        "startup_time_seconds": i + 1
                    }
            
            # Se non si è avviato entro 10 secondi
            return {
                "success": False,
                "message": "Timeout: Ollama server non si è avviato entro 10 secondi",
                "error": "startup_timeout"
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "message": "Ollama non trovato. Assicurati che sia installato e nel PATH",
                "error": "ollama_not_found",
                "install_guide": self._get_install_guide()
            }
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Errore nell'avvio del server: {str(e)}",
                "error": str(e)
            }
    
    async def stop_server(self) -> Dict[str, Any]:
        """Ferma il server Ollama"""
        try:
            status = await self.get_server_status()
            
            if status.status != "running":
                return {
                    "success": True,
                    "message": "Ollama server non è in esecuzione",
                    "already_stopped": True
                }
            
            if status.pid is None:
                return {
                    "success": False,
                    "message": "Impossibile trovare il PID del processo Ollama",
                    "error": "pid_not_found"
                }
            
            # Termina il processo
            try:
                process = psutil.Process(status.pid)
                process.terminate()
                
                # Attendi che il processo termini (max 5 secondi)
                for i in range(5):
                    await asyncio.sleep(1)
                    if not process.is_running():
                        return {
                            "success": True,
                            "message": "Ollama server fermato con successo",
                            "shutdown_time_seconds": i + 1
                        }
                
                # Se non si è fermato, forza la terminazione
                process.kill()
                await asyncio.sleep(1)
                
                if not process.is_running():
                    return {
                        "success": True,
                        "message": "Ollama server fermato forzatamente",
                        "forced_shutdown": True
                    }
                else:
                    return {
                        "success": False,
                        "message": "Impossibile fermare il server Ollama",
                        "error": "shutdown_failed"
                    }
                    
            except psutil.NoSuchProcess:
                return {
                    "success": True,
                    "message": "Ollama server già fermato",
                    "already_stopped": True
                }
                
        except Exception as e:
            self.logger.error(f"Error stopping Ollama server: {e}")
            return {
                "success": False,
                "message": f"Errore nella terminazione del server: {str(e)}",
                "error": str(e)
            }
    
    async def restart_server(self) -> Dict[str, Any]:
        """Riavvia il server Ollama"""
        try:
            # Prima ferma il server
            stop_result = await self.stop_server()
            
            if not stop_result["success"] and not stop_result.get("already_stopped", False):
                return {
                    "success": False,
                    "message": "Impossibile fermare il server per il riavvio",
                    "error": stop_result.get("error", "stop_failed")
                }
            
            # Attendi un momento prima di riavviare
            await asyncio.sleep(2)
            
            # Poi riavvia
            start_result = await self.start_server()
            
            if start_result["success"]:
                return {
                    "success": True,
                    "message": "Ollama server riavviato con successo",
                    "restart_completed": True,
                    "new_pid": start_result.get("pid")
                }
            else:
                return {
                    "success": False,
                    "message": "Riavvio fallito: impossibile riavviare il server",
                    "error": start_result.get("error", "restart_failed")
                }
                
        except Exception as e:
            self.logger.error(f"Error restarting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Errore nel riavvio del server: {str(e)}",
                "error": str(e)
            }
    
    def _get_install_guide(self) -> Dict[str, str]:
        """Restituisce guida installazione per l'OS corrente"""
        os_name = platform.system()
        
        if os_name == "Windows":
            return {
                "download_url": "https://ollama.com/download/windows",
                "instructions": "1. Scarica Ollama per Windows dal link sopra\n2. Esegui l'installer\n3. Riavvia il terminale\n4. Digita: ollama serve"
            }
        elif os_name == "Darwin":  # macOS
            return {
                "download_url": "https://ollama.com/download/mac",
                "instructions": "1. Scarica Ollama per macOS dal link sopra\n2. Trascina Ollama nella cartella Applicazioni\n3. Apri Terminal e digita: ollama serve"
            }
        else:  # Linux
            return {
                "download_url": "https://ollama.com/download/linux",
                "instructions": "1. Esegui: curl -fsSL https://ollama.com/install.sh | sh\n2. Oppure scarica manualmente dal link sopra\n3. Avvia con: ollama serve"
            }
    
    def format_uptime(self, seconds: Optional[int]) -> str:
        """Formatta uptime in modo leggibile"""
        if seconds is None:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    async def get_diagnostic_info(self) -> Dict[str, Any]:
        """Ottieni informazioni diagnostiche complete"""
        diagnostic = {
            "ollama_installed": False,
            "ollama_in_path": False,
            "server_running": False,
            "port_accessible": False,
            "system_resources": {},
            "recommendations": []
        }
        
        try:
            # Verifica se Ollama è installato
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                diagnostic["ollama_installed"] = True
                diagnostic["ollama_in_path"] = True
                diagnostic["ollama_version"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            diagnostic["recommendations"].append("Installare Ollama dal sito ufficiale")
        
        # Verifica status server
        status = await self.get_server_status()
        diagnostic["server_running"] = status.status == "running"
        
        if status.status == "running":
            diagnostic["server_info"] = {
                "pid": status.pid,
                "uptime": self.format_uptime(status.uptime_seconds),
                "memory_mb": status.memory_mb,
                "cpu_percent": status.cpu_percent
            }
        else:
            diagnostic["recommendations"].append("Avviare Ollama server con: ollama serve")
        
        # Informazioni sistema
        try:
            diagnostic["system_resources"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        except Exception:
            pass
        
        return diagnostic
