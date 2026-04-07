"""
JMatPro Worker Pool Manager

This module provides a fault-tolerant parallel worker system for JMatPro calculations.
Workers are initialized at startup and wait for tasks. Failed tasks are automatically
reassigned to ensure all calculations complete.
"""

import os
import sys
import time
import copy
import multiprocessing
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import traceback


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Represents a thermodynamic calculation task"""
    task_id: int
    composition: List[float]
    elements: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None  # raw: phase_name -> {molar_fraction, elements, composition} or float
    error: Optional[str] = None
    worker_id: Optional[int] = None
    start_time: Optional[float] = None
    retry_count: int = 0


def _worker_process(worker_id: int, task_queue: multiprocessing.Queue, 
                    result_queue: multiprocessing.Queue, 
                    config: Dict[str, Any]):
    """
    Worker process that runs JMatPro calculations.
    
    Args:
        worker_id: Unique identifier for this worker
        task_queue: Queue to receive tasks from
        result_queue: Queue to send results to
        config: Configuration dictionary with JMatPro settings
    """

    # Setup JMatPro imports in worker process
    ROOT_PATH = config.get('root_path')
    DLL_PATH = config.get('dll_path')
    
    if DLL_PATH and os.path.exists(DLL_PATH):
        os.chdir(DLL_PATH)
        os.add_dll_directory(os.getcwd())
        if ROOT_PATH:
            sys.path.append(ROOT_PATH)
    
    try:
        from apiwrapper import (
            JMP_MATERIAL_NICKEL_BASED_SUPERALLOY,
            JMP_COMPOSITION_UNIT_ATOMIC_PERCENT,
            JMP_TEMPERATURE_UNIT_CELSIUS,
            JMP_SOLVER_CALCULATION_SINGLE_POINT,
            JMP_THERMODYNAMIC_PROPERTY_MOLAR_FRACTION,
            jmpSetMaterialType,
            jmpSetAlloyElements,
            jmpSetCompositionUnit,
            jmpSetAlloyComposition,
            jmpSetSolverCalculationType,
            jmpSetTemperatureUnit,
            jmpSetSolverTemperature,
            jmpSetDefaultPhases,
            jmpRunSolverCalculation,
            jmpGetFoundPhases,
            jmpGetThermodynamicPropertyAt,
            jmpGetPhaseCompositionAt,
            jmpSetWorkingDirectory,
            JMPError,
            jmpSetScreenOutput
        )
    except ImportError as e:
        # Send error back to master
        result_queue.put({
            'worker_id': worker_id,
            'error': f"Failed to import JMatPro API: {e}",
            'fatal': True
        })
        return
    
    # Extract configuration (use defaults when value is None, since config is built with None placeholders)
    material_type = config.get('material_type') or JMP_MATERIAL_NICKEL_BASED_SUPERALLOY
    unit = config.get('unit') or JMP_COMPOSITION_UNIT_ATOMIC_PERCENT
    unit_temperature = config.get('unit_temperature') or JMP_TEMPERATURE_UNIT_CELSIUS
    calculation_type = config.get('calculation_type') or JMP_SOLVER_CALCULATION_SINGLE_POINT
    temperature = config.get('temperature', 1100.0)

    # Give this worker its own temp folder so JMatPro output is not overwritten by other workers
    base_temp = os.path.join(os.path.abspath(DLL_PATH.rstrip(os.sep)), 'temp')
    worker_temp = os.path.join(base_temp, f'worker_{worker_id}{os.sep}')
    try:
        os.makedirs(worker_temp, exist_ok=True)
        jmpSetWorkingDirectory(worker_temp)
    except Exception as e:
        result_queue.put({
            'worker_id': worker_id,
            'error': f"Failed to set worker temp dir: {e}",
            'fatal': True
        })
        return

    jmpSetMaterialType(material_type)
    jmpSetCompositionUnit(unit)
    jmpSetMaterialType(material_type)
    jmpSetSolverCalculationType(calculation_type)
    jmpSetTemperatureUnit(unit_temperature)
    jmpSetSolverTemperature(temperature)
    jmpSetScreenOutput(0)

    # Signal worker is ready
    result_queue.put({
        'worker_id': worker_id,
        'status': 'ready'
    })

    # Per-worker cache: (composition_key, elements_tuple) -> result dict. Key uses rounded composition for stability.
    _cache = {}
    _cache_round_digits = 3

    # Main worker loop
    while True:
        try:
            # Get task from queue (blocking with timeout to allow periodic health checks)
            try:
                task_data = task_queue.get(timeout=0.005)
            except:
                continue
            
            # Check for termination signal
            if task_data is None:
                break
            
            task_id = task_data['task_id']
            composition = task_data['composition']
            elements = task_data['elements']
            
            start_time = time.time()
            composition = list(composition)
            elements = list(elements)
            composition_key = tuple(round(float(x), _cache_round_digits) for x in composition)
            elements_key = tuple(elements)
            cache_key = (composition_key, elements_key)

            try:
                # Return cached result if present
                if cache_key in _cache:
                    result_queue.put({
                        'worker_id': worker_id,
                        'task_id': task_id,
                        'status': 'completed',
                        'result': _cache[cache_key],
                        'duration': time.time() - start_time,
                        'cached': True,
                    })
                    continue

                # Set up JMatPro calculation
                # jmpSetMaterialType(material_type)
                jmpSetAlloyElements(elements)
                # jmpSetCompositionUnit(unit)
                jmpSetAlloyComposition(composition)
                # jmpSetSolverCalculationType(calculation_type)
                # jmpSetTemperatureUnit(unit_temperature)
                # jmpSetSolverTemperature(temperature)
                jmpSetDefaultPhases()
                
                # Run calculation with timeout check
                jmpRunSolverCalculation()
                
                # Get found phases; return molar_fraction and phase composition per phase
                try:
                    found_phases = jmpGetFoundPhases()
                except JMPError:
                    found_phases = []
                
                def get_phase_fraction(phase_name):
                    if phase_name not in found_phases:
                        return 0.0
                    try:
                        f = jmpGetThermodynamicPropertyAt(
                            JMP_THERMODYNAMIC_PROPERTY_MOLAR_FRACTION,
                            phase_name,
                            temperature
                        )
                        return max(0.0, f)
                    except (JMPError, ValueError, TypeError):
                        return 0.0
                
                def get_phase_composition(phase_name):
                    try:
                        el_list, comp_list = jmpGetPhaseCompositionAt(phase_name, temperature)
                        return list(el_list), list(comp_list)
                    except (JMPError, ValueError, TypeError):
                        return [], []
                
                result = {}
                for p in found_phases:
                    frac = get_phase_fraction(p)
                    elements_phase, composition_phase = get_phase_composition(p)
                    sum_non_ox = 0.0
                    for elem, comp in zip(elements_phase, composition_phase):
                        sum_non_ox += comp if elem not in ["O", "N", "Ni", "H", "Fe"] else 0.0

                    result[p] = {
                        "molar_fraction": frac,
                        "elements": elements_phase,
                        "composition": composition_phase,
                        "sum_non_ox": sum_non_ox,
                    }
 
                _cache[cache_key] = result
                result_queue.put({
                    'worker_id': worker_id,
                    'task_id': task_id,
                    'status': 'completed',
                    'result': result,
                    'duration': time.time() - start_time
                })
                
            except JMPError as e:
                # JMatPro calculation error
                result_queue.put({
                    'worker_id': worker_id,
                    'task_id': task_id,
                    'status': 'failed',
                    'error': str(e),
                    'duration': time.time() - start_time
                })
            except Exception as e:
                # Unexpected error
                result_queue.put({
                    'worker_id': worker_id,
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f"{type(e).__name__}: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'duration': time.time() - start_time
                })
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Critical error in worker
            result_queue.put({
                'worker_id': worker_id,
                'error': f"Worker critical error: {type(e).__name__}: {str(e)}",
                'fatal': True
            })
            break
    
    # Signal worker termination
    result_queue.put({
        'worker_id': worker_id,
        'status': 'terminated'
    })


class JMatProWorkerPool:
    """
    Manages a pool of JMatPro worker processes with fault tolerance.
    
    Features:
    - Persistent workers that wait for tasks
    - Automatic task retry on failure
    - Timeout detection for unresponsive workers
    - Guaranteed task completion
    """
    
    def __init__(self, num_workers: int = None, 
                 temperature: float = 1100.0,
                 task_timeout: float = 300.0,
                 max_retries: int = 3):
        """
        Initialize worker pool. Workers return raw JMatPro output (phase_name -> molar_fraction).
        Elements are provided from the outside when submitting tasks (submit_tasks(..., elements=)).
        """
        if num_workers is None:
            num_workers = max(1, os.cpu_count() or 1)
        
        self.num_workers = num_workers
        self.temperature = temperature
        self.task_timeout = task_timeout
        self.max_retries = max_retries
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.abspath(os.path.join(script_dir, '..'))
        self.dll_path = os.path.join(self.root_path, "thermodynamics", "Runtime", "x64") + os.sep
        
        # Worker management
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self.worker_status = {}  # Track worker health
        
        # Task tracking
        self.tasks = {}  # task_id -> Task
        self.next_task_id = 0
        self.pending_tasks = []  # Queue of pending task IDs

        # Fast wake when a result arrives (set by monitor, waited on in get_results)
        import threading
        self._result_event = threading.Event()
        self._results_lock = threading.Lock()
        
        # Initialize workers
        self._start_workers()
        
        # Start result monitor thread
        self.monitor_thread = None
        self.running = True
        
    def _start_workers(self):
        """Start all worker processes"""
        config = {
            'root_path': self.root_path,
            'dll_path': self.dll_path,
            'material_type': None,
            'unit': None,
            'unit_temperature': None,
            'calculation_type': None,
            'temperature': self.temperature,
            'task_timeout': self.task_timeout
        }
        
        for i in range(self.num_workers):
            worker = multiprocessing.Process(
                target=_worker_process,
                args=(i, self.task_queue, self.result_queue, config),
                daemon=False
            )
            worker.start()
            self.workers.append(worker)
            self.worker_status[i] = {
                'process': worker,
                'alive': True,
                'current_task': None,
                'task_start_time': None,
                'last_heartbeat': time.time()
            }
    
    def _monitor_results(self):
        """Monitor result queue and handle completed/failed tasks"""
        while self.running:
            try:
                # Check result queue; short timeout for responsive result processing
                try:
                    result = self.result_queue.get(timeout=0.005)
                except:
                    continue
                
                worker_id = result.get('worker_id')
                
                # Handle worker control messages (ready/terminated); completed/failed fall through
                if 'status' in result:
                    if result['status'] == 'ready':
                        self.worker_status[worker_id]['last_heartbeat'] = time.time()
                        continue
                    if result['status'] == 'terminated':
                        self.worker_status[worker_id]['alive'] = False
                        continue
                
                # Handle fatal errors
                if result.get('fatal', False):
                    print(f"Fatal error in worker {worker_id}: {result.get('error', 'Unknown')}")
                    self.worker_status[worker_id]['alive'] = False
                    # Restart worker
                    self._restart_worker(worker_id)
                    continue
                
                # Handle task completion (one at a time; deep copy so no shared refs between tasks)
                task_id = result.get('task_id')
                if task_id is not None and task_id in self.tasks:
                    res = result.get('result')
                    res_copy = copy.deepcopy(res) if res is not None else {}
                    with self._results_lock:
                        task = self.tasks[task_id]
                        if result['status'] == 'completed':
                            task.status = TaskStatus.COMPLETED
                            task.result = res_copy
                            task.worker_id = None
                            task.start_time = None
                            self.worker_status[worker_id]['current_task'] = None
                            self.worker_status[worker_id]['task_start_time'] = None
                            self._result_event.set()
                        elif result['status'] == 'failed':
                            task.status = TaskStatus.FAILED
                            task.error = result.get('error', 'Unknown error')
                            task.worker_id = None
                            task.start_time = None
                            self.worker_status[worker_id]['current_task'] = None
                            self.worker_status[worker_id]['task_start_time'] = None
                            self._result_event.set()
                            if task.retry_count < self.max_retries:
                                task.retry_count += 1
                                task.status = TaskStatus.PENDING
                                self.pending_tasks.append(task_id)
                            else:
                                print(f"Task {task_id} failed after {self.max_retries} retries: {task.error}")
                
            except Exception as e:
                print(f"Error in result monitor: {e}")
                import traceback
                traceback.print_exc()
    
    def _check_timeouts(self):
        """Check for tasks that have exceeded timeout (by worker or by task start_time when queue flooded)."""
        current_time = time.time()
        # Timeout by worker (when worker had been assigned)
        for worker_id, status in self.worker_status.items():
            if status['current_task'] is not None:
                task_id = status['current_task']
                elapsed = current_time - status['task_start_time']
                
                if elapsed > self.task_timeout:
                    task = self.tasks.get(task_id)
                    if task and task.status == TaskStatus.RUNNING:
                        print(f"Task {task_id} timed out on worker {worker_id} after {elapsed:.1f}s")
                        task.status = TaskStatus.TIMEOUT
                        task.worker_id = None
                        task.start_time = None
                        status['current_task'] = None
                        status['task_start_time'] = None
                        if task.retry_count < self.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.PENDING
                            self.pending_tasks.append(task_id)
                        else:
                            print(f"Task {task_id} timed out after {self.max_retries} retries")
                        if not status['process'].is_alive():
                            self._restart_worker(worker_id)
        # Timeout by task start_time (flooded queue: no worker assigned yet or worker_id is None)
        for task_id, task in list(self.tasks.items()):
            if task.status == TaskStatus.RUNNING and task.start_time is not None:
                elapsed = current_time - task.start_time
                if elapsed > self.task_timeout:
                    print(f"Task {task_id} timed out (no worker) after {elapsed:.1f}s")
                    task.status = TaskStatus.TIMEOUT
                    task.start_time = None
                    task.worker_id = None
                    if task.retry_count < self.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        self.pending_tasks.append(task_id)
                    else:
                        print(f"Task {task_id} timed out after {self.max_retries} retries")
    
    def _restart_worker(self, worker_id: int):
        """Restart a failed worker"""
        print(f"Restarting worker {worker_id}...")
        
        # Terminate old worker if still alive
        old_worker = self.worker_status[worker_id]['process']
        if old_worker.is_alive():
            old_worker.terminate()
            old_worker.join(timeout=5.0)
            if old_worker.is_alive():
                old_worker.kill()
        
        # Reassign any running task
        if self.worker_status[worker_id]['current_task'] is not None:
            task_id = self.worker_status[worker_id]['current_task']
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.PENDING
                task.worker_id = None
                task.start_time = None
                self.pending_tasks.append(task_id)
        
        # Start new worker
        config = {
            'root_path': self.root_path,
            'dll_path': self.dll_path,
            'material_type': None,
            'unit': None,
            'unit_temperature': None,
            'calculation_type': None,
            'temperature': self.temperature,
            'task_timeout': self.task_timeout
        }
        
        new_worker = multiprocessing.Process(
            target=_worker_process,
            args=(worker_id, self.task_queue, self.result_queue, config),
            daemon=False
        )
        new_worker.start()
        
        self.worker_status[worker_id] = {
            'process': new_worker,
            'alive': True,
            'current_task': None,
            'task_start_time': None,
            'last_heartbeat': time.time()
        }
    
    def _distribute_all_tasks(self):
        """Put all pending tasks on the queue at once so workers never wait for the main thread."""
        while self.pending_tasks:
            task_id = self.pending_tasks.pop(0)
            task = self.tasks[task_id]
            try:
                self.task_queue.put({
                    'task_id': task_id,
                    'composition': list(task.composition),
                    'elements': list(task.elements)
                }, timeout=1.0)
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                task.worker_id = None  # unknown until result comes back
            except Exception as e:
                print(f"Error queuing task {task_id}: {e}")
                self.pending_tasks.insert(0, task_id)
                break
    
    def submit_tasks(self, compositions: List[List[float]], 
                    elements: List[str]) -> List[int]:
        """
        Submit multiple calculation tasks.
        
        Args:
            compositions: List of composition lists
            elements: Element symbols (provided from outside, e.g. from CA/config)
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for composition in compositions:
            task_id = self.next_task_id
            self.next_task_id += 1
            
            task = Task(
                task_id=task_id,
                composition=list(composition),
                elements=list(elements)
            )
            
            self.tasks[task_id] = task
            self.pending_tasks.append(task_id)
            task_ids.append(task_id)
        
        return task_ids
    
    def get_results(self, task_ids: List[int], 
                   wait: bool = True, 
                   timeout: Optional[float] = None) -> Dict[int, Dict[str, float]]:
        """
        Get results for submitted tasks.
        
        Returns:
            Dictionary mapping task_id to raw result (phase_name -> molar_fraction)
        """
        start_time = time.time()
        
        # Start monitor thread if not running
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            import threading
            self.monitor_thread = threading.Thread(target=self._monitor_results, daemon=True)
            self.monitor_thread.start()
        
        # Flood queue with all tasks immediately so workers never wait for the main thread
        self._distribute_all_tasks()
        
        results = {}
        
        while wait:
            # Check timeouts (including RUNNING tasks when queue was flooded)
            self._check_timeouts()
            # Put any retried tasks back on the queue
            if self.pending_tasks:
                self._distribute_all_tasks()
            
            # Check if all tasks are completed (lock so we don't read while monitor is writing)
            all_completed = True
            with self._results_lock:
                for task_id in task_ids:
                    task = self.tasks.get(task_id)
                    if task is None:
                        continue
                    if task.status == TaskStatus.COMPLETED:
                        results[task_id] = task.result
                    elif task.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                        results[task_id] = {}
                    else:
                        all_completed = False
            
            if all_completed:
                break
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                with self._results_lock:
                    for task_id in task_ids:
                        if task_id not in results:
                            task = self.tasks.get(task_id)
                            if task and task.status == TaskStatus.COMPLETED:
                                results[task_id] = task.result
                            else:
                                results[task_id] = {}
                break
            
            # Wake quickly when monitor processes a result; short max wait to re-check timeouts
            self._result_event.clear()
            self._result_event.wait(timeout=0.05)
        
        return results
    
    def shutdown(self):
        """Shutdown worker pool"""
        self.running = False
        
        # Send termination signals to workers
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, timeout=1.0)
            except:
                pass
        
        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
                if worker.is_alive():
                    worker.kill()
        
        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)


if __name__ == '__main__':
    # Test the worker pool
    print("Testing JMatProWorkerPool...")
    
    pool = JMatProWorkerPool(num_workers=1, temperature=1100.0)
    
    # Submit test tasks (elements provided from outside)
    compositions = [
        [58.0, 1.0, 1.0, 40.0],
        [60.0, 3.0, 3.0, 34.0]
    ]
    elements = ["Ni", "Cr", "Al", "O"]
    task_ids = pool.submit_tasks(compositions, elements=elements)
    print(f"Submitted {len(task_ids)} tasks")
    
    # Get results
    results = pool.get_results(task_ids, wait=True, timeout=10.0)
    
    for task_id, result in results.items():
        print(f"\n--- Task {task_id} ---")
        if not isinstance(result, dict):
            print(f"  result: {result}")
            continue
        for phase_name, data in result.items():
            if isinstance(data, dict):
                frac = data.get("molar_fraction", 0.0)
                el = data.get("elements", [])
                comp = data.get("composition", [])
                print(f"  {phase_name}: molar_fraction = {frac}")
                if el and comp:
                    comp_str = ", ".join(f"{e}={v:.4f}" for e, v in zip(el, comp))
                    print(f"    composition: {comp_str}")
            else:
                print(f"  {phase_name}: {data}")
    
    pool.shutdown()
    print("\nTest completed")
