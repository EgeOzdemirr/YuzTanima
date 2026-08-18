import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

from aiohttp import web


def get_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


ROOT = get_root()
PERSONS_ROOT = ROOT / "public" / "gallery" / "persons"
GALLERY_CONFIG = ROOT / "services" / "gallery_builder" / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gallery_ui_server")

RUN_LOCK = asyncio.Lock()


@web.middleware
async def no_cache_static(request: web.Request, handler):
    response = await handler(request)
    if request.path.startswith(("/ui", "/data/events")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def safe_person_name(name: str) -> str:
    if not name:
        return "person"
    name = name.strip().strip(". ")
    name = re.sub(r'[<>:"/\\\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "person"


async def save_upload(file_field) -> Path | None:
    filename = Path(file_field.filename).name
    stem = safe_person_name(Path(filename).stem)
    ext = Path(filename).suffix or ".jpg"
    person_dir = PERSONS_ROOT / stem
    person_dir.mkdir(parents=True, exist_ok=True)
    dest = person_dir / f"photo{ext}"
    written = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file_field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    if written == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        logger.warning("Skipped empty upload for %s", filename)
        return None
    return dest


async def run_script(args: list) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    stdout = (out or b"").decode("utf-8", errors="ignore")
    stderr = (err or b"").decode("utf-8", errors="ignore")
    return proc.returncode, stdout, stderr


async def handle_import(request: web.Request) -> web.Response:
    reader = await request.multipart()
    with3d = False
    saved = []
    async for part in reader:
        if part.name == "with3d":
            value = (await part.text()).strip().lower()
            with3d = value in {"1", "true", "yes", "on"}
            continue
        if part.name == "files":
            saved_path = await save_upload(part)
            if saved_path:
                saved.append(str(saved_path))

    if not saved:
        return web.json_response({"ok": False, "error": "no_files"}, status=400)

    async with RUN_LOCK:
        steps = []
        if with3d:
            steps.append(
                [
                    sys.executable,
                    str(ROOT / "services" / "gallery_builder" / "build_3d_gallery.py"),
                    "--config",
                    str(GALLERY_CONFIG),
                ]
            )
        steps.append(
            [
                sys.executable,
                str(ROOT / "services" / "gallery_builder" / "build_gallery.py"),
                "--config",
                str(GALLERY_CONFIG),
            ]
        )
        logs = []
        for cmd in steps:
            code, out, err = await run_script(cmd)
            logs.append({"cmd": " ".join(cmd), "code": code, "stdout": out[-4000:], "stderr": err[-4000:]})
            if code != 0:
                return web.json_response({"ok": False, "error": "build_failed", "logs": logs}, status=500)

    return web.json_response({"ok": True, "saved": saved, "with3d": with3d})


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    app = web.Application(middlewares=[no_cache_static])
    app.router.add_route("POST", "/api/gallery/import", handle_import)
    app.router.add_route("GET", "/health", handle_health)
    app.router.add_static("/ui", str(ROOT / "ui"), show_index=True)
    app.router.add_static("/public", str(ROOT / "public"))
    app.router.add_static("/data", str(ROOT / "data"))
    app.router.add_static("/services", str(ROOT / "services"))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gallery UI server with import API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
