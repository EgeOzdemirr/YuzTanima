import ctypes
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path


HOST = "127.0.0.1"
UI_PORT = 9000
UI_URL = f"http://{HOST}:{UI_PORT}/ui/index.html"
HEALTH_URL = f"http://{HOST}:{UI_PORT}/health"


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJob:
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Bu baslatici Windows icin hazirlandi.")

        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self.kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.kernel32.SetInformationJobObject.restype = ctypes.c_int
        self.kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_int

        self.handle = self.kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self.kernel32.SetInformationJobObject(
            self.handle,
            self.JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, proc: subprocess.Popen) -> None:
        ok = self.kernel32.AssignProcessToJobObject(self.handle, ctypes.c_void_p(int(proc._handle)))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


def print_header() -> None:
    print("Yuz Tanima baslatiliyor...")
    print("Bu pencere acik kaldigi surece UI ve kamera servisi calisir.")
    print("Kapatmak icin bu pencereyi kapatin veya Ctrl+C basin.")
    print()


def wait_for_enter() -> None:
    try:
        input("Pencereyi kapatmak icin Enter'a basin...")
    except EOFError:
        pass


def ensure_runtime_files(root: Path) -> None:
    for relative in [
        "data/events",
        "data/gallery",
        "public/captures",
        "public/gallery/persons",
        "logs",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)

    copies = [
        ("data/events/latest_suspicious.example.json", "data/events/latest_suspicious.json"),
        ("data/gallery/gallery.example.json", "data/gallery/gallery.json"),
    ]
    for source, dest in copies:
        source_path = root / source
        dest_path = root / dest
        if not dest_path.exists() and source_path.exists():
            shutil.copyfile(source_path, dest_path)


def add_nvidia_dll_paths(root: Path, env: dict[str, str]) -> None:
    nvidia_root = root / "3ddfav3" / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return
    bin_dirs = [str(path) for path in nvidia_root.glob("*/bin") if path.is_dir()]
    if not bin_dirs:
        return
    env["PATH"] = os.pathsep.join(bin_dirs + [env.get("PATH", "")])


def port_is_busy(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_health(timeout_seconds: int = 30) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start_service(
    *,
    name: str,
    args: list[str],
    root: Path,
    log_path: Path,
    env: dict[str, str],
    job: WindowsJob,
    log_files: list,
) -> subprocess.Popen:
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_files.append(log_file)
    log_file.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {name}\n")
    log_file.write(" ".join(args) + "\n")

    proc = subprocess.Popen(
        args,
        cwd=str(root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        job.assign(proc)
    except Exception:
        proc.terminate()
        raise

    print(f"{name} basladi. PID={proc.pid} Log={log_path}")
    return proc


def stop_processes(job: WindowsJob | None, processes: list[tuple[str, subprocess.Popen]]) -> None:
    print()
    print("Servisler kapatiliyor...")

    if job is not None:
        job.close()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(proc.poll() is not None for _, proc in processes):
            return
        time.sleep(0.2)

    for _, proc in processes:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def run() -> int:
    # PyInstaller ile derlenmiş EXE'de sys.frozen=True olur;
    # EXE proje kök dizinine yerleştirildiğinden parent klasör = proje kökü.
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parents[1]
    python = root / "3ddfav3" / "Scripts" / "python.exe"
    if not python.exists():
        raise RuntimeError(f"Python ortami bulunamadi: {python}")

    ensure_runtime_files(root)

    if port_is_busy(HOST, UI_PORT):
        raise RuntimeError(
            f"{UI_PORT} portu zaten dolu. Eski bir Baslat penceresi aciksa kapatin ve tekrar deneyin."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = root / "logs"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    add_nvidia_dll_paths(root, env)

    job = WindowsJob()
    log_files: list = []
    processes: list[tuple[str, subprocess.Popen]] = []

    try:
        ui_proc = start_service(
            name="UI sunucusu",
            args=[
                str(python),
                str(root / "services" / "gallery_builder" / "gallery_ui_server.py"),
                "--host",
                HOST,
                "--port",
                str(UI_PORT),
            ],
            root=root,
            log_path=logs_dir / f"windows_ui_{timestamp}.log",
            env=env,
            job=job,
            log_files=log_files,
        )
        processes.append(("UI sunucusu", ui_proc))

        if not wait_for_health():
            raise RuntimeError("UI sunucusu hazir olmadi. Log dosyasini kontrol edin.")

        camera_proc = start_service(
            name="Kamera servisi",
            args=[
                str(python),
                str(root / "services" / "camera_runtime" / "run_camera.py"),
                "--config",
                str(root / "services" / "camera_runtime" / "config.yaml"),
            ],
            root=root,
            log_path=logs_dir / f"windows_camera_{timestamp}.log",
            env=env,
            job=job,
            log_files=log_files,
        )
        processes.append(("Kamera servisi", camera_proc))

        time.sleep(2)
        for name, proc in processes:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} erken kapandi. Log dosyasini kontrol edin.")

        print()
        print(f"UI hazir: {UI_URL}")
        webbrowser.open(UI_URL)
        print("Tarayici kapatilsa bile servisler bu pencere kapanana kadar calisir.")
        print()

        while True:
            for name, proc in processes:
                return_code = proc.poll()
                if return_code is not None:
                    raise RuntimeError(f"{name} kapandi. Cikis kodu: {return_code}")
            time.sleep(1)
    finally:
        stop_processes(job, processes)
        for log_file in log_files:
            try:
                log_file.close()
            except Exception:
                pass


def main() -> int:
    print_header()
    try:
        return run()
    except KeyboardInterrupt:
        print()
        print("Kapatma istegi alindi.")
        return 0
    except Exception as exc:
        print()
        print(f"HATA: {exc}")
        print()
        wait_for_enter()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
