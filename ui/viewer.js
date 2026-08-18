import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export class GLBViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.canvas.dataset.loaded = "0";
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f2436);
    this.camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
    this.camera.position.set(0, 0.15, 3.5);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    // color management for brighter, accurate rendering
    if (this.renderer.outputColorSpace !== undefined) {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else {
      this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.9;
    this.loader = new GLTFLoader();
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.1, 0);
    this.lookTarget = { yaw: 0, pitch: 0 };
    this.lookCurrent = { yaw: 0, pitch: 0 };
    this.clock = new THREE.Clock();

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(0.8, 1.2, 1);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.9);
    fillLight.position.set(-0.8, 0.4, 1);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.7);
    rimLight.position.set(0, 0.6, -1.2);
    const hemi = new THREE.HemisphereLight(0xf4f7ff, 0x243247, 0.7);
    const ambient = new THREE.AmbientLight(0xc4cfdd, 1.1);

    this.scene.add(keyLight, fillLight, rimLight, hemi, ambient);
    this.current = null;
    this.loadingUrl = null;
    this.loadedUrl = null;
    this._loadToken = 0;
    this._abortController = null;
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _disposeCurrent() {
    if (this.current) {
      this.scene.remove(this.current);
      this.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
      });
    }
    this.current = null;
  }

  clear() {
    this.cancelLoad(true);
  }

  cancelLoad(clearScene = true) {
    this._loadToken += 1;
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
    this.loadingUrl = null;
    if (clearScene) {
      this._disposeCurrent();
      this.loadedUrl = null;
    }
    this.canvas.dataset.loaded = "0";
  }

  _resourcePath(url) {
    const clean = (url || "").split("?")[0];
    const idx = clean.lastIndexOf("/");
    return idx >= 0 ? clean.slice(0, idx + 1) : "/";
  }

  load(url) {
    if (!url) {
      this.clear();
      return;
    }
    this.cancelLoad(true);
    const token = this._loadToken;
    this.loadingUrl = url;
    const controller = new AbortController();
    this._abortController = controller;
    const bustUrl = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
    fetch(bustUrl, { signal: controller.signal, cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then(
        (buffer) =>
          new Promise((resolve, reject) => {
            this.loader.parse(buffer, this._resourcePath(url), resolve, reject);
          })
      )
      .then((gltf) => {
        if (token !== this._loadToken) return;
        this._abortController = null;
        this.loadingUrl = null;
        this._disposeCurrent();
        this.current = gltf.scene;
        // Start facing camera and keep fixed
        this.current.rotation.set(0, 0, 0);
        this.current.traverse((child) => {
          if (child.isMesh && child.material) {
            child.material.side = THREE.FrontSide;
            child.material.needsUpdate = true;
          }
        });
        this.scene.add(this.current);
        this._fitToView();
        this.loadedUrl = url;
        this.canvas.dataset.loaded = "1";
      })
      .catch((err) => {
        if (token !== this._loadToken) return;
        this._abortController = null;
        this.loadingUrl = null;
        if (err?.name === "AbortError") return;
        console.error("Failed to load GLB", err);
        this._disposeCurrent();
        this.loadedUrl = null;
        this.canvas.dataset.loaded = "0";
      });
  }

  setLookOffset(normX, normY) {
    // normX, normY in [-1,1]
    this.lookTarget.yaw = normX * 0.4; // yaw (y axis)
    this.lookTarget.pitch = normY * 0.25; // pitch (x axis)
  }

  _fitToView() {
    if (!this.current) return;
    const box = new THREE.Box3().setFromObject(this.current);
    if (!isFinite(box.min.x) || !isFinite(box.max.x)) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    // Center the model
    this.current.position.sub(center);

    const maxDim = Math.max(size.x, size.y, size.z);
    if (!maxDim || maxDim <= 0) return;

    const aspect = this.renderer.domElement.clientWidth / this.renderer.domElement.clientHeight || 1;
    const fov = (this.camera.fov * Math.PI) / 180;
    const fitHeightDistance = maxDim / (2 * Math.tan(fov / 2));
    const fitWidthDistance = maxDim / (2 * Math.tan(fov / 2) * aspect);
    const distance = 1.4 * Math.max(fitHeightDistance, fitWidthDistance);

    this.camera.position.set(0, 0, distance);
    this.camera.near = distance / 100;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  _animate() {
    requestAnimationFrame(this._animate);
    if (this.current) {
      // smooth look towards target
      this.lookCurrent.yaw += (this.lookTarget.yaw - this.lookCurrent.yaw) * 0.08;
      this.lookCurrent.pitch += (this.lookTarget.pitch - this.lookCurrent.pitch) * 0.08;
      // Gentle idle motion while keeping the face front-facing
      const t = this.clock.getElapsedTime();
      const autoYaw = Math.sin(t * 0.4) * 0.26;
      const autoPitch = Math.sin(t * 0.3) * 0.1;
      const totalYaw = Math.max(-0.3, Math.min(0.3, autoYaw + this.lookCurrent.yaw));
      const totalPitch = Math.max(-0.16, Math.min(0.16, autoPitch + this.lookCurrent.pitch));
      this.current.rotation.y = totalYaw;
      this.current.rotation.x = totalPitch;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
