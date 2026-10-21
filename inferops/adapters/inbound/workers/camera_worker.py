"""An assigned camera shard with an authenticated, owner-local HTTP endpoint."""
import os
import threading
import time


def start_watchdog(application, *, timeout_s=60, exit_process=os._exit):
    """A native thread can terminate a hung worker whose event loop cannot exit."""
    stop = threading.Event()
    if timeout_s <= 0:
        raise ValueError("worker watchdog timeout must be positive")

    def watch():
        while not stop.wait(min(1, timeout_s / 4)):
            if time.monotonic() - application.last_ownership_renewal > timeout_s:
                exit_process(70)
                return
    thread = threading.Thread(target=watch, name="camera-worker-watchdog", daemon=True)
    thread.start()
    return stop, thread


def run_camera_worker(manifest, worker_id, *, host="0.0.0.0", port=8080):
    from inferops.adapters.outbound.persistence.worker_manifest import load_worker_manifest
    spec = load_worker_manifest(manifest)
    if worker_id not in {worker.worker_id for worker in spec.workers}:
        raise ValueError("worker ID is not present in the manifest")
    os.environ["INFEROPS_ROLE"] = "camera-worker"
    os.environ["INFEROPS_WORKER_ID"] = worker_id
    os.environ["INFEROPS_WORKER_MANIFEST"] = str(manifest)
    import uvicorn

    from inferops.adapters.inbound.http.app import create_application
    uvicorn.run(create_application(), host=host, port=port)
