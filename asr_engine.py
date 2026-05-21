from importlib import metadata
from pathlib import Path
import ctypes
import os
import site
import subprocess
import sys

import numpy as np

_DLL_DIRECTORY_HANDLES = []
_PRELOADED_DLLS = []


def _prepare_nvidia_dll_paths():
    candidates = []
    for root in [Path(sys.prefix) / "Lib" / "site-packages"]:
        if root.exists():
            candidates.append(root)
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    dll_dirs = []
    for root in candidates:
        sherpa_lib = root / "sherpa_onnx" / "lib"
        if sherpa_lib.exists():
            dll_dirs.append(sherpa_lib)
        nvidia_dir = root / "nvidia"
        if not nvidia_dir.exists():
            continue
        dll_dirs.extend(path for path in nvidia_dir.glob("*/bin") if path.is_dir())

    for dll_dir in dict.fromkeys(str(path) for path in dll_dirs):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir))
            except OSError:
                pass
    _preload_cuda_dlls(dll_dirs)


def _preload_cuda_dlls(dll_dirs):
    names = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
    ]
    for dll_dir in dll_dirs:
        for name in names:
            dll_path = dll_dir / name
            if not dll_path.exists():
                continue
            try:
                _PRELOADED_DLLS.append(ctypes.WinDLL(str(dll_path)))
            except OSError:
                pass


_prepare_nvidia_dll_paths()

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime probe
    ort = None
else:
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        except Exception:
            pass

import sherpa_onnx


class SherpaStreamingASR:
    def __init__(self, model_dir: Path, num_threads: int = 2, provider: str = "auto"):
        self.model_dir = Path(model_dir)
        self.sample_rate = 16000
        self.model_path = self._resolve_model_path()
        self.tokens_path = self.model_dir / "tokens.txt"
        self._check_files()
        self.provider = "cpu"
        self.provider_message = ""
        self.recognizer = self._create_recognizer(num_threads=num_threads, provider=provider)

    def _resolve_model_path(self):
        for name in ("model.fp16.onnx", "model.int8.onnx", "model.onnx"):
            path = self.model_dir / name
            if path.is_file():
                return path
        return self.model_dir / "model.fp16.onnx"

    def _create_recognizer(self, num_threads: int, provider: str):
        providers = self._provider_candidates(provider)
        errors = []
        for candidate in providers:
            if candidate == "cuda" and not self._is_probe_child():
                ok, message = self._probe_cuda_recognizer()
                if not ok:
                    errors.append(f"cuda probe failed: {message}")
                    continue
            try:
                recognizer = self._build_recognizer(num_threads, candidate)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            self.provider = candidate
            self.provider_message = "; ".join(errors)
            return recognizer
        raise RuntimeError("Failed to create sherpa-onnx recognizer. " + " | ".join(errors))

    def _provider_candidates(self, provider: str):
        if provider != "auto":
            return [provider]
        return ["cuda", "cpu"] if self._cuda_build_available() else ["cpu"]

    def _cuda_build_available(self) -> bool:
        if "+cuda" in self._sherpa_version().lower():
            return True
        if ort is None:
            return False
        try:
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _sherpa_version(self) -> str:
        version = getattr(sherpa_onnx, "__version__", "")
        if version:
            return version
        try:
            return metadata.version("sherpa-onnx")
        except metadata.PackageNotFoundError:
            return ""

    def _is_probe_child(self) -> bool:
        return os.environ.get("SHERPA_ONNX_CUDA_PROBE_CHILD") == "1"

    def _probe_cuda_recognizer(self):
        code = (
            "from asr_engine import SherpaStreamingASR; "
            f"SherpaStreamingASR(r'{self.model_dir}', provider='cuda'); "
            "print('cuda-ok', flush=True)"
        )
        env = os.environ.copy()
        env["SHERPA_ONNX_CUDA_PROBE_CHILD"] = "1"
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(self.model_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as exc:
            return False, str(exc)
        if completed.returncode == 0:
            return True, ""
        message = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
        return False, message

    def _build_recognizer(self, num_threads: int, provider: str):
        return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(self.tokens_path),
            model=str(self.model_path),
            num_threads=num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.2,
            rule2_min_trailing_silence=0.6,
            rule3_min_utterance_length=10.0,
            provider=provider,
            decoding_method="greedy_search",
        )

    def _check_files(self):
        missing = [p.name for p in (self.model_path, self.tokens_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing ASR model files: " + ", ".join(missing))

    def create_stream(self):
        return self.recognizer.create_stream()

    def accept_audio(self, stream, samples: np.ndarray) -> str:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if hasattr(stream, "accept_waveform"):
            stream.accept_waveform(self.sample_rate, samples)
        else:
            self.recognizer.accept_waveform(stream, self.sample_rate, samples)
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        if hasattr(result, "text"):
            return result.text.strip()
        return str(result).strip()

    def is_endpoint(self, stream) -> bool:
        return self.recognizer.is_endpoint(stream)

    def reset(self, stream):
        self.recognizer.reset(stream)
