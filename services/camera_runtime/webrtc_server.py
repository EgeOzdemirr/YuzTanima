import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

# Ensure repo root is importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_runtime.pipeline import CameraPipeline

logger = logging.getLogger("webrtc_server")

pcs = set()
pipeline: Optional[CameraPipeline] = None


class FrameHub:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0

    def update(self, frame) -> None:
        with self._lock:
            self._frame = frame
            self._sequence += 1
        self._loop.call_soon_threadsafe(self._event.set)

    async def get(self, last_sequence: Optional[int] = None):
        while True:
            with self._lock:
                frame = self._frame
                sequence = self._sequence
            if frame is not None and sequence != last_sequence:
                return frame, sequence
            await self._event.wait()
            self._event.clear()


class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, hub: FrameHub) -> None:
        super().__init__()
        self._hub = hub
        self._last_sequence: Optional[int] = None

    async def recv(self) -> VideoFrame:
        frame, self._last_sequence = await self._hub.get(self._last_sequence)
        video = VideoFrame.from_ndarray(frame, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video.pts = pts
        video.time_base = time_base
        return video


async def wait_for_ice_complete(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return

    fut = asyncio.get_running_loop().create_future()

    @pc.on("icegatheringstatechange")
    def on_ice_state_change() -> None:
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(None)

    await fut


def capture_loop(hub: FrameHub, stop_event: threading.Event) -> None:
    if pipeline is None:
        return
    show_preview = bool(getattr(pipeline, "preview_enabled", False))
    window_name = getattr(pipeline, "preview_window", "OpenCV Camera")
    ui_cfg = getattr(pipeline, "ui_cfg", {}) or {}
    try:
        live_stream_fps = float(ui_cfg.get("live_stream_fps", 0) or 0)
    except (TypeError, ValueError):
        live_stream_fps = 0.0
    min_publish_interval = 1.0 / live_stream_fps if live_stream_fps > 0 else 0.0
    last_publish_at = 0.0

    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        decode_cfg = pipeline.camera_cfg.get("decode", {})
        width = int(decode_cfg.get("width") or 0)
        height = int(decode_cfg.get("height") or 0)
        if width > 0 and height > 0:
            cv2.resizeWindow(window_name, width, height)
    try:
        for frame in pipeline._frame_generator(stop_event=stop_event):
            processed = pipeline.process_frame(frame)
            now = time.monotonic()
            if min_publish_interval <= 0 or now - last_publish_at >= min_publish_interval:
                hub.update(processed)
                last_publish_at = now
            if show_preview:
                preview = pipeline._resize_for_preview(processed)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
    finally:
        if show_preview:
            cv2.destroyWindow(window_name)


@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    if not response.prepared:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


async def handle_offer(request: web.Request) -> web.Response:
    try:
        params = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    if "sdp" not in params or "type" not in params:
        return web.json_response({"error": "missing sdp/type"}, status=400)

    hub: FrameHub = request.app["frame_hub"]
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info("PeerConnection created: %s", id(pc))

    pc.addTrack(CameraVideoTrack(hub))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await pc.close()
            pcs.discard(pc)

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_for_ice_complete(pc)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_mjpeg(request: web.Request) -> web.StreamResponse:
    hub: FrameHub = request.app["frame_hub"]
    ui_cfg = getattr(pipeline, "ui_cfg", {}) if pipeline is not None else {}
    try:
        quality = int(ui_cfg.get("live_stream_jpeg_quality", 65) or 65)
    except (TypeError, ValueError):
        quality = 65
    quality = max(35, min(90, quality))

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    last_sequence: Optional[int] = None
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    try:
        while True:
            frame, last_sequence = await hub.get(last_sequence)
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            data = encoded.tobytes()
            await response.write(b"--frame\r\n")
            await response.write(b"Content-Type: image/jpeg\r\n")
            await response.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
            await response.write(data)
            await response.write(b"\r\n")
    except asyncio.CancelledError:
        raise
    except (ConnectionResetError, BrokenPipeError):
        logger.info("MJPEG client disconnected")
    return response


async def on_startup(app: web.Application) -> None:
    loop = asyncio.get_running_loop()
    hub = FrameHub(loop)
    stop_event = threading.Event()
    app["frame_hub"] = hub
    app["stop_event"] = stop_event
    thread = threading.Thread(target=capture_loop, args=(hub, stop_event), daemon=True)
    app["capture_thread"] = thread
    thread.start()


async def on_shutdown(app: web.Application) -> None:
    stop_event = app.get("stop_event")
    if stop_event:
        stop_event.set()
    await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
    pcs.clear()


def create_app(config_path: str) -> web.Application:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    global pipeline
    pipeline = CameraPipeline(config_path)
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_route("POST", "/offer", handle_offer)
    app.router.add_route("OPTIONS", "/offer", handle_offer)
    app.router.add_route("GET", "/health", handle_health)
    app.router.add_route("GET", "/stream.mjpg", handle_mjpeg)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def run_server(config_path: str, host: str = "0.0.0.0", port: int = 8081) -> None:
    app = create_app(config_path)
    web.run_app(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTC camera broadcast server")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    args = parser.parse_args()
    run_server(args.config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
