import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_runtime.pipeline import CameraPipeline
from services.camera_runtime.webrtc_server import run_server


def load_config(config_path: str):
    try:
        import yaml
    except Exception:
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run camera runtime loop")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional)")
    parser.add_argument("--webrtc", action="store_true", help="Run WebRTC server to stream OpenCV camera to UI")
    parser.add_argument("--webrtc-host", default=None, help="WebRTC bind host")
    parser.add_argument("--webrtc-port", type=int, default=None, help="WebRTC bind port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.config:
        config_path = args.config
    else:
        base_dir = None
        if getattr(sys, "frozen", False):
            base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        else:
            base_dir = ROOT
        default_candidates = [
            base_dir / "services" / "camera_runtime" / "config.yaml",
            ROOT / "services" / "camera_runtime" / "config.yaml",
        ]
        config_path = next((str(p) for p in default_candidates if p.exists()), str(default_candidates[0]))

    cfg = load_config(config_path)
    webrtc_cfg = cfg.get("camera", {}).get("webrtc", {})
    use_webrtc = args.webrtc or bool(webrtc_cfg)
    if use_webrtc:
        host = args.webrtc_host or webrtc_cfg.get("host") or "0.0.0.0"
        port = args.webrtc_port or int(webrtc_cfg.get("port") or 8081)
        run_server(config_path, host=host, port=port)
        return

    pipeline = CameraPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
