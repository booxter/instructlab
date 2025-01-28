# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from typing import Callable
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid

# Third Party
# TODO: where is this coming from? include in requirements
from filelock import FileLock
import psutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import ILAB_PROCESS_MODES, ILAB_PROCESS_STATUS

logger = logging.getLogger(__name__)


# TODO: may want to add more type hints for args etc.
# TODO: consider adding super() calls to init methods - it's debatable though
# TODO: may be prudent to adopt psutil.Process class and then use its interfaces like .kill and .terminate

# TODO: are log files ever cleaned up? Should they be cleaned up?
# TODO: Is it possible to inspect completed processes in the cli (with their logs)?
# TODO: is it possible to attach and detach without killing process? (upd: no, it's not possible, but it should be)


class ProcessRegistry:
    def __init__(self):
        self.processes = {}

    def add_process(
        # TODO: don't reuse type keyword as var name
        # TODO: add type hints
        self, local_uuid, pid, children_pids, type, log_file, start_time, status
    ):
        # TODO: introduce typed dict for this?
        self.processes[str(local_uuid)] = {
            "pid": pid,
            "children_pids": children_pids,
            "type": type,
            "log_file": log_file,
            # TODO: why do we transform this to isoformat and back?
            "start_time": datetime.strptime(
                start_time, "%Y-%m-%d %H:%M:%S"
            ).isoformat(),
            "status": status,
        }

    # TODO: reuse add_process for this
    def load_entry(self, key, value):
        self.processes[key] = value


def load_registry() -> ProcessRegistry:
    process_registry = ProcessRegistry()
    lock_path = DEFAULTS.PROCESS_REGISTRY_LOCK_FILE
    # TODO: check if these file locks are "hard" enough for the use
    # TODO: nit: move the lock thing slightly below closer to where it's used to avoid potential miss on the lock __exit__
    # TODO: so what happens when timeout is up? will this raise? can I execute two parallel clients at the same time?
    # TODO: do we need to lock for reading the same way as we do for writing? do we need RO lock vs RW lock depending on the goal?
    lock = FileLock(lock_path, timeout=1)
    # TODO: nit: docstrings should not be used for comments. I believe python will unnecessarily instantiate a string. Plus it's weird.
    """Load the process registry from a file, if it exists."""
    # we do not want a persistent registry in memory. This causes issues when in scenarios where you switch registry files (ex, in a unit test, or with multiple users)
    # but the registry with incorrect processes still exists in memory.
    with lock:
        # TODO:
        # if not exists:
        #    return process_registry
        # read
        if os.path.exists(DEFAULTS.PROCESS_REGISTRY_FILE):
            # TODO: nit: r is default mode for open
            with open(DEFAULTS.PROCESS_REGISTRY_FILE, "r") as f:
                # TODO: handle malformed json?
                data = json.load(f)
                for key, value in data.items():
                    process_registry.load_entry(key=key, value=value)
        else:
            logger.debug("No existing process registry found. Starting fresh.")
    # TODO: at this point, lock is no more. If I now execute another instance
    # of the client, will it read the "old" state and potentially race with the
    # first client instance when writing the new registry state back to disc?
    return process_registry


def save_registry(process_registry):
    """Save the current process registry to a file."""
    lock_path = DEFAULTS.PROCESS_REGISTRY_LOCK_FILE
    lock = FileLock(lock_path, timeout=1)
    with lock, open(DEFAULTS.PROCESS_REGISTRY_FILE, "w") as f:
        json.dump(dict(process_registry.processes), f)


class Tee:
    # TODO: nit: debatable but docstrings here are pretty redundant; remove?
    def __init__(self, log_file):
        """
        Initialize a Tee object.

        Args:
            # TODO: probably a file object, not a path string?
            log_file (str): Path to the log file where the output should be written.
        """
        self.log_file = log_file # TODO: nit: remove?
        self.terminal = sys.stdout
        self.log = log_file  # Line-buffered

    def write(self, message):
        """
        Write the message to both the terminal and the log file.

        Args:
            message (str): The message to write.
        """
        # TODO: should the message be encoded to utf-8 here?
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Ensure all data is written to the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """
        Close the log file.
        """
        if self.log: # TODO: when is it not?
            self.log.close()


def format_command(
    # TODO: what is kwargs and why
    target: Callable, extra_imports: list[tuple[str, ...]], **kwargs
) -> str:
    """
    Formats a command given the target and any extra python imports to add

    Args:
        target: Callable
        extra_imports: list[tuple[str, ...]]
    Returns:
        cmd: str
    """
    # Prepare the subprocess command string
    cmd = (
        f"import {target.__module__}; {target.__module__}.{target.__name__}(**{kwargs})"
    )

    # Handle extra imports (if any)
    if extra_imports:
        import_statements = "\n".join(
            [f"from {imp[0]} import {', '.join(imp[1:])}" for imp in extra_imports]
        )
        cmd = f"{import_statements}\n{cmd}"
    return cmd


# TODO: add type hint for log arg
# TODO: return empty list not None when no children? Or just a single tuple with children being 2+ entries?
def start_process(cmd: str, log) -> tuple[int | None, list[int] | None]:
    """
    Starts a subprocess and captures PID and Children PIDs

    Args:
        cmd: str
        log: _FILE

    Returns:
        pid: int
        children_pids: list[int]
    """
    children_pids = []
    p = subprocess.Popen(
        # TODO: need to think through if this is always the right way to find
        # python to use; perhaps we should have proper console_scripts for each
        # action and then trigger these? they would definitely have a proper
        # python set.
        ["python", "-c", cmd], # TODO: use python3
        # TODO: check if all these are correct / needed
        universal_newlines=True, # TODO: nit: redundant. we already set text= (which is preferred)
        text=True,
        stdout=log,
        stderr=log, # TODO: I wonder if we should actually reuse the same file to keep both. Need to think.
        start_new_session=True, # TODO: I think this is posix only, so no win32. are we ok with it?
        encoding="utf-8", # TODO: does it assume something it shouldn't about the current locale? (what if it's not utf-8?)
        bufsize=1,  # Line-buffered for real-time output
    )
    # TODO: why sleep here?
    time.sleep(1)
    # we need to get all of the children processes spawned
    # to be safe, we will need to try and kill all of the ones which still exist when the user wants us to
    # however, representing this to the user is difficult. So let's track the parent pid and associate the children with it in the registry

    # TODO: check what this spinning does and why; oh perhaps that's because
    # children may take some time to start? I think we may want to instead
    # enumerate children when we kill them, not at the start. Even 5 seconds
    # may be not enough for some processes to start all of their children. Or
    # maybe some processes in the future will spawn more and more children as
    # they proceed. We can't assume that only initial 1+2.5 seconds are enough
    # to start all children.
    max_retries = 5
    retry_interval = 0.5  # seconds
    parent = psutil.Process(p.pid)
    for _ in range(max_retries):
        children = parent.children(recursive=True)
        # TODO: so are we spinning until SOME children are up, then breaking? How do we know that these are all the children there will be?
        if children:
            for child in children:
                children_pids.append(child.pid)
            break
        time.sleep(retry_interval)
    else:
        logger.debug("No child processes detected. Tracking parent process.")
    # Check if subprocess was successfully started
    # TODO: should we first check that parent is up before walking through its children? (move this check up?)
    if p.poll() is not None:
        logger.warning(f"Process {p.pid} failed to start.")
        # TODO: use raise to indicate error, don't overload return values
        return None, None  # Process didn't start
    return p.pid, children_pids


def add_process(
    process_mode: str,  # TODO: make it an enum
    process_type: str,  # TODO: make it an enum
    target: Callable,
    extra_imports: list[tuple[str, ...]],  # TODO: check this type is fitting
    **kwargs,  # TODO: this should not be a catch-all -> split into separate args
):
    """
    Start a detached process using subprocess.Popen, logging its output.

    Args:
        process_mode (str): Mode we are running in, Detached or Attached.
        process_type (str): Type of process, ex: Generation.
        target (func): The target function to kick off in the subprocess or to run in the foreground.
        extra_imports (list[tuple(str...)]): a list of the extra imports to splice into the python subprocess command.

    Returns:
        None # TODO: why not return the process info and / or status code?
    """
    process_registry = load_registry()
    if target is None:
        # TODO: this is unreachable, apparently; or maybe type hint is wrong and it can be None?
        return None, None

    # TODO: why do we need a uuid when there's pid already? is it so that UI is
    # more "neat"? Is it neat with uuids though? Maybe it should be a
    # process-name + some postfix?
    local_uuid = uuid.uuid1() # TODO: is the returned uuid safe? is it important?
    log_file = None

    log_dir = os.path.join(DEFAULTS.LOGS_DIR, process_type.lower())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{process_type.lower()}-{local_uuid}.log")
    pid: int | None = os.getpid()
    children_pids: list[int] | None = []
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kwargs["log_file"] = log_file
    kwargs["local_uuid"] = str(local_uuid)
    kwargs["process_mode"] = process_mode
    # TODO: maybe instead of having a different mode for attached / detached,
    # always start with detached, then attach to it. Then we don't need modes,
    # and whatever we implement for attaching (e.g. ctrl+c or ctrl+d behaviors)
    # would work exactly the same way for either mode.
    if process_mode == ILAB_PROCESS_MODES.DETACHED:
        # TODO: should probably remove all these asserts from production code;
        # also, do any of these reveal actual issues / gaps in coverage / code
        # / type hints? Check.
        assert isinstance(log_file, str)
        cmd = format_command(target=target, extra_imports=extra_imports, **kwargs)
        # Open the subprocess in the background, redirecting stdout and stderr to the log file
        with open(log_file, "a+") as log:
            pid, children_pids = start_process(cmd=cmd, log=log)
            if pid is None or children_pids is None:
                # process didn't start
                # TODO: so how do we know that it failed? should we know?
                return None # TODO: nit: ...which is just 'return'
            assert isinstance(pid, int) and isinstance(children_pids, list)
    # Add the process info to the shared registry
    process_registry.add_process(
        local_uuid=local_uuid,
        pid=pid,
        children_pids=children_pids,
        type=process_type,
        log_file=log_file,
        start_time=start_time_str,
        status=ILAB_PROCESS_STATUS.RUNNING,
    )
    # TODO: should probably not dump this, or perhaps if this is indeed
    # important to know which log file was used (why?), then expose it via cli
    # output when listing processes? (upd: ok maybe the reason why we do this
    # is because it's impossible to exit from re-attached process w/o killing
    # it. we should fix it.)
    logger.info(
        f"Started subprocess with PID {pid}. Logs are being written to {log_file}."
    )
    save_registry(
        process_registry=process_registry
    )  # Persist registry after adding process
    if process_mode == ILAB_PROCESS_MODES.ATTACHED:
        with open(log_file, "a+") as log:
            # TODO: nit = sys.stdout = sys.stderr = Tee(log)
            sys.stdout = Tee(log)
            sys.stderr = sys.stdout
            try:
                # TODO: why do we pass here all the kwargs that belong to DETACHED case? log_file etc. Target is not required to accept these, right?
                target(**kwargs)  # Call the function
            finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def all_processes_running(pids: list[int]) -> bool:
    """
    Returns if a process and all of its children are still running
    Args:
        pids (list): a list of all PIDs to check
    """
    return all(psutil.pid_exists(pid) for pid in pids)


# TODO: add remove arg description
def stop_process(local_uuid, remove=True):
    """
    Stop a running process.

    Args:
        local_uuid (str): uuid of the process to stop.
    """
    process_registry = load_registry()
    # we should kill the parent process, and also children processes.
    pid = process_registry.processes[local_uuid]["pid"]
    children_pids = process_registry.processes[local_uuid]["children_pids"]
    all_processes = [pid] + children_pids
    # TODO: should we start with children instead? otherwise if e.g. we kill
    # the parent, then something fails, then children may live on I think.
    for process in all_processes:
        try:
            # TODO: should we actually -9 right away? it's not graceful
            os.kill(process, signal.SIGKILL)
            logger.info(f"Process {process} terminated.")
        # TODO: so when does it happen that we receive PermissionError? haven't we ourselves started the process?
        except (ProcessLookupError, PermissionError):
            logger.warning(
                f"Process {process} was not running or could not be stopped."
            )
    if remove:
        process_registry.processes.pop(local_uuid, None)
    else:
        # since we just killed the processes, we cannot depend on it to update itself, mark as done and set end time
        process_registry.processes[local_uuid]["status"] = ILAB_PROCESS_STATUS.DONE
        process_registry.processes[local_uuid]["end_time"] = datetime.now().isoformat()
    # TODO: so if kill succeeded but save failed, then do we end up with inconsistent state? is it handled somehow?
    save_registry(process_registry=process_registry)


def update_status(local_uuid, status):
    """
    Updates the status of a process.

    Args:
        local_uuid (str): uuid of the process to stop.
    """

    process_registry = load_registry()
    entry = process_registry.processes.get(str(local_uuid), {})
    if entry:
        process_registry.processes[str(local_uuid)]["end_time"] = (
            datetime.now().isoformat()
        )
        process_registry.processes[str(local_uuid)]["status"] = status
        save_registry(process_registry=process_registry)


def list_processes():
    """
    # TODO: split out concerns. marking is a different operation from listing.
    Constructs a list of processes and their statuses. Marks processes as ready for removal if necessary

    # TODO: nit: remove useless args / returns description below
    Args:
        None

    Returns:
        None
    """
    process_registry = load_registry()
    if not process_registry:
        # TODO: this will never be true because load_registry will always
        # return a ProcessRegistry instance, which is always truthy
        logger.info("No processes currently in the registry.")
        return

    list_of_processes = []

    processes_to_remove = []
    for local_uuid, entry in process_registry.processes.items():
        # assume all processes are running and not ready for removal unless status indicates otherwise
        status = entry.get("status", "")
        all_pids = [entry["pid"]] + entry["children_pids"]
        if not all_processes_running(all_pids):
            # if all of our child or parent processes are not running, we should either a. remove this process, or b. mark it ready for removal after this list
            # TODO: what if some are still running and some don't?..
            if status in (ILAB_PROCESS_STATUS.DONE, ILAB_PROCESS_STATUS.ERRORED):
                # TODO: should we remove after listing? perhaps we should have
                # a way to retrieve the state for completed processes that is
                # more long term? what happens with artifacts like logs?

                # if this has been marked as done remove it after listing once
                # but, we cannot remove while looping as that will cause errors.
                processes_to_remove.append(local_uuid)

            # if not, list it, but mark it as ready for removal
            # TODO: ^ where is this marking as ready for removal happening?
        now = datetime.now()
        start_time = datetime.fromisoformat(entry.get("start_time"))
        # Calculate runtime
        runtime = now - start_time
        if "end_time" in entry:
            runtime = datetime.fromisoformat(entry.get("end_time")) - start_time
        # Convert timedelta to a human-readable string (HH:MM:SS)
        # TODO: runtime is already timedelta object; we can just .hours and .minutes and .seconds as needed
        total_seconds = int(runtime.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # TODO: so no process ever takes more than 99 hours? should we assume it here? (the format string will truncate and e.g. show 100:00:00 as 00:00:00)
        runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        list_of_processes.append(
            (
                entry.get("type"),
                entry.get("pid"),
                local_uuid,
                entry.get("log_file"),
                runtime_str,
                status,
            )
        )

    # we want to stop and remove from the process registry only after we have listed it
    # this allows completed processes to be seen in the list once before being removed from the registry

    # TODO: so are we killing a parent when e.g. some of its children are no
    # longer running? this may be premature to do I think. We should let parent
    # complete on its own.
    for proc in processes_to_remove:
        stop_process(local_uuid=proc, remove=True)

    return list_of_processes


# TODO: return status code?
def attach_process(local_uuid: str): # TODO: I think uuids have a specific type in python - UUID/SafeUUID? -, should we use it?
    """
    Attach to a running process and display its output in real-time.

    Args:
        local_uuid (str): UUID of the process to attach to
    """
    process_registry = load_registry()
    if local_uuid not in process_registry.processes:
        logger.warning("Process not found.")
        return

    process_info = process_registry.processes[local_uuid]
    log_file = process_info["log_file"]

    # TODO: can it ever happen? don't we open the file before starting the process (passing the file object to the process through popen interface)?
    if not os.path.exists(log_file):
        logger.warning(
            "Log file not found. The process may not have started logging yet."
        )
        return # TODO: make it an error detectable by the cli user

    # TODO: we should have a way to detach without killing the process
    logger.info(f"Attaching to process {local_uuid}. Press Ctrl+C to detach and kill.")
    all_pids = [process_info["pid"]] + process_info["children_pids"]
    if not all_processes_running(all_pids):
        # TODO: so if some processes are not running, we should not attach? why?
        # at least tell the user this happens maybe? (looks like we'll return 0 as if all went well?)
        return
    try:
        with open(log_file, "a+") as log:
            # TODO: hm, will this potentially break lines? (look for first \n?)
            log.seek(0, os.SEEK_END)  # Move to the end of the log file
            while all_processes_running(all_pids):
                line = log.readline()
                # Check for non-empty and non-whitespace-only lines
                # TODO: why would we skip empty lines? they may be part of a useful output (e.g. some table padding added for readability?)
                if line.strip():
                    print(line.strip())
                else:
                    # TODO: switch to blocking read if this is not how it already works (as the sleep suggests?)

                    # TODO: why would we wait if there's more stuff to print? What if the process produce more output than 10 lines per
                    # second? will be back up and not print the latest as quickly as it is produced?
                    time.sleep(0.1)  # Wait briefly before trying again
    # TODO: is it the only exception that can happen here?
    except KeyboardInterrupt:
        logger.info("\nDetaching from and killing process.")
    finally:
        # TODO: why would we kill it? allow to detach
        stop_process(local_uuid=local_uuid, remove=False)


def get_latest_process() -> str | None:
    """
    Returns the last process added to the registry to quickly allow users to attach to it.

    Returns:
        last_key (str): a string UUID to attach to
    """
    process_registry = load_registry()
    keys = process_registry.processes.keys()
    # no processes
    if len(keys) == 0:
        return None
    # TODO: do we re-read keys here because it's iterator and it was exhausted? we should capture it once then as list(process_registry.processes.keys()) and then work with it
    # TODO: it's a dictionary, its keys are NOT sorted. This will pick a RANDOM process, not the latest. We should sort by the stored timestamp in the registry entries.
    last_key = list(process_registry.processes.keys())[-1]
    # TODO: don't assert in code - maybe should be forced by mypy
    assert isinstance(last_key, str)
    return last_key
