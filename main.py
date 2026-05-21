import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

from asr_engine import SherpaStreamingASR
from command_parser import CommandParser

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - shown in GUI
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc
else:
    SOUNDDEVICE_IMPORT_ERROR = None

try:
    from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyQt5 is not installed. Install dependencies with: pip install -r requirements.txt"
    ) from exc
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WAKE_WORD = "迈克同志"
DEFAULT_STOP_WORD = "over"
VOICE_RMS_THRESHOLD = 0.01
SILENCE_SECONDS = 3.0

MIC_OFF = "MIC_OFF"
WAIT_WAKE = "WAIT_WAKE"
ACTIVE_COMMAND = "ACTIVE_COMMAND"


class RecognizerWorker(QObject):
    status_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    live_text_changed = pyqtSignal(str)
    command_started = pyqtSignal()
    command_finalized = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        device_index=None,
        wake_word=DEFAULT_WAKE_WORD,
        stop_word=DEFAULT_STOP_WORD,
    ):
        super().__init__()
        self.device_index = device_index
        self._wake_word = wake_word.strip() or DEFAULT_WAKE_WORD
        self._stop_word = stop_word.strip() or DEFAULT_STOP_WORD
        self._parser = CommandParser()
        self._queue = queue.Queue()
        self._running = False
        self._shutdown_requested = False
        self._manual_start_requested = False
        self._manual_stop_requested = False
        self._mode = WAIT_WAKE
        self._initial_command_text = ""
        self._stream_command_text = ""
        self._last_voice_time = None
        self._provider_label = "ASR"

    def request_start(self, wake_word=None, stop_word=None):
        if wake_word is not None:
            self.set_wake_word(wake_word)
        if stop_word is not None:
            self.set_stop_word(stop_word)
        self._manual_start_requested = True

    def request_stop(self):
        self._manual_stop_requested = True

    def request_shutdown(self):
        self._shutdown_requested = True

    def set_wake_word(self, wake_word):
        wake_word = (wake_word or "").strip()
        self._wake_word = wake_word or DEFAULT_WAKE_WORD

    def set_stop_word(self, stop_word):
        stop_word = (stop_word or "").strip()
        self._stop_word = stop_word or DEFAULT_STOP_WORD

    def run(self):
        if sd is None:
            self.error_occurred.emit(f"sounddevice failed to import: {SOUNDDEVICE_IMPORT_ERROR}")
            self.finished.emit()
            return

        try:
            self.status_changed.emit("正在加载 ASR 模型...")
            asr = SherpaStreamingASR(PROJECT_DIR, provider="auto")
            self._provider_label = "GPU/CUDA" if asr.provider == "cuda" else "CPU"
            if asr.provider == "cpu" and asr.provider_message:
                self._provider_label = "CPU（CUDA 初始化失败，已回退）"
            stream_state = asr.create_stream()
            sample_rate = asr.sample_rate
            block_size = int(sample_rate * 0.1)
            self._running = True
            self._mode = WAIT_WAKE
            self.mode_changed.emit(WAIT_WAKE)
            self.status_changed.emit(f"等待唤醒（ASR: {self._provider_label}）")

            def audio_callback(indata, frames, time_info, status):
                if status:
                    self.status_changed.emit(str(status))
                self._queue.put(indata[:, 0].astype(np.float32).copy())

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=self.device_index,
                callback=audio_callback,
            ):
                while self._running:
                    if self._shutdown_requested:
                        if self._mode == ACTIVE_COMMAND:
                            self._finalize_command(asr, stream_state)
                        break

                    if self._manual_start_requested:
                        self._manual_start_requested = False
                        self._begin_command(asr, stream_state, "")

                    if self._manual_stop_requested:
                        self._manual_stop_requested = False
                        if self._mode == ACTIVE_COMMAND:
                            self._finalize_command(asr, stream_state)

                    try:
                        chunk = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        if self._mode == ACTIVE_COMMAND and self._is_silent_timed_out():
                            self._finalize_command(asr, stream_state)
                        continue

                    text = asr.accept_audio(stream_state, chunk)
                    if self._mode == WAIT_WAKE:
                        self._handle_wait_wake(asr, stream_state, text)
                    elif self._mode == ACTIVE_COMMAND:
                        self._handle_active_command(asr, stream_state, text, chunk)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self._running = False
            self.mode_changed.emit(MIC_OFF)
            self.finished.emit()

    def _handle_wait_wake(self, asr, stream_state, text):
        if text:
            self.live_text_changed.emit(text)
        if self._parser.contains_wake_word(text, self._wake_word):
            initial_text = self._parser.extract_command_text(text, self._wake_word)
            self._begin_command(asr, stream_state, initial_text)
        elif asr.is_endpoint(stream_state):
            asr.reset(stream_state)

    def _handle_active_command(self, asr, stream_state, text, chunk):
        if self._has_voice(chunk):
            self._last_voice_time = time.monotonic()
        if text:
            self._stream_command_text = text
        combined = self._combined_command_text()
        if self._parser.contains_control_word(combined, self._stop_word):
            command_text = self._parser.strip_control_word(combined, self._stop_word)
            self._initial_command_text = command_text
            self._stream_command_text = ""
            self.live_text_changed.emit(command_text)
            self._finalize_command(asr, stream_state)
            return
        self.live_text_changed.emit(combined)
        if self._is_silent_timed_out():
            self._finalize_command(asr, stream_state)

    def _begin_command(self, asr, stream_state, initial_text):
        if self._mode == ACTIVE_COMMAND:
            return
        asr.reset(stream_state)
        self._mode = ACTIVE_COMMAND
        self._initial_command_text = initial_text.strip()
        self._stream_command_text = ""
        self._last_voice_time = time.monotonic()
        self.command_started.emit()
        self.mode_changed.emit(ACTIVE_COMMAND)
        self.status_changed.emit("识别中")
        self.live_text_changed.emit(self._initial_command_text)

    def _finalize_command(self, asr, stream_state):
        command_text = self._combined_command_text()
        if self._parser.contains_control_word(command_text, self._stop_word):
            command_text = self._parser.strip_control_word(command_text, self._stop_word)
        result = self._parser.parse(command_text, self._wake_word)
        self.command_finalized.emit(result)
        self._mode = WAIT_WAKE
        self._initial_command_text = ""
        self._stream_command_text = ""
        self._last_voice_time = None
        asr.reset(stream_state)
        self.mode_changed.emit(WAIT_WAKE)
        self.status_changed.emit(f"等待唤醒（ASR: {self._provider_label}）")

    def _combined_command_text(self):
        return f"{self._initial_command_text}{self._stream_command_text}".strip()

    def _has_voice(self, samples):
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms >= VOICE_RMS_THRESHOLD

    def _is_silent_timed_out(self):
        return (
            self._mode == ACTIVE_COMMAND
            and self._last_voice_time is not None
            and time.monotonic() - self._last_voice_time >= SILENCE_SECONDS
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.mode = MIC_OFF
        self.current_result = None
        self.current_result_archived = True
        self.history_items = []
        self.setWindowTitle("语音指令识别系统")
        self.resize(1080, 720)
        self._build_ui()
        self._load_devices()
        self._set_mode(MIC_OFF)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(12)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        root.addLayout(toolbar)

        self.mic_button = QPushButton("开启")
        self.mic_button.clicked.connect(self.toggle_microphone)
        toolbar.addWidget(self.mic_button)

        self.start_button = QPushButton("开始识别")
        self.start_button.clicked.connect(self.start_command)
        toolbar.addWidget(self.start_button)

        self.stop_button = QPushButton("停止识别")
        self.stop_button.clicked.connect(self.stop_command)
        toolbar.addWidget(self.stop_button)

        self.status_label = QLabel("状态：未开启")
        self.status_label.setObjectName("status")
        toolbar.addWidget(self.status_label, 1)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        toolbar.addWidget(self.device_combo)

        wake_label = QLabel("wake_word")
        toolbar.addWidget(wake_label)

        self.wake_word_edit = QLineEdit(DEFAULT_WAKE_WORD)
        self.wake_word_edit.setMinimumWidth(170)
        self.wake_word_edit.textChanged.connect(self.on_wake_word_changed)
        toolbar.addWidget(self.wake_word_edit)

        stop_label = QLabel("stop_word")
        toolbar.addWidget(stop_label)

        self.stop_word_edit = QLineEdit(DEFAULT_STOP_WORD)
        self.stop_word_edit.setMinimumWidth(120)
        self.stop_word_edit.textChanged.connect(self.on_stop_word_changed)
        toolbar.addWidget(self.stop_word_edit)

        self.live_text = QPlainTextEdit()
        self.live_text.setReadOnly(True)
        self.live_text.setPlaceholderText("实时识别文字会显示在这里")
        self.live_text.setMinimumHeight(190)
        root.addWidget(self._wrap_group("实时识别文字", self.live_text), 2)

        result_group = QGroupBox("识别结果")
        result_layout = QHBoxLayout(result_group)
        result_layout.setContentsMargins(10, 18, 10, 10)
        result_layout.setSpacing(10)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("停止识别后，结构化 JSON 会显示在这里")
        self.result_text.setMinimumHeight(280)
        result_layout.addWidget(self.result_text, 5)

        history_shell = QWidget()
        history_layout = QVBoxLayout(history_shell)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(8)

        history_title = QLabel("历史任务")
        history_title.setAlignment(Qt.AlignCenter)
        history_title.setObjectName("historyTitle")
        history_layout.addWidget(history_title)

        self.history_content = QWidget()
        self.history_list = QVBoxLayout(self.history_content)
        self.history_list.setContentsMargins(4, 4, 4, 4)
        self.history_list.setSpacing(8)
        self.history_list.addStretch(1)

        history_scroll = QScrollArea()
        history_scroll.setWidgetResizable(True)
        history_scroll.setWidget(self.history_content)
        history_scroll.setMinimumWidth(150)
        history_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        history_layout.addWidget(history_scroll, 1)
        result_layout.addWidget(history_shell)

        root.addWidget(result_group, 3)

        self.setStyleSheet(
            """
            QWidget { font-family: 'Microsoft YaHei', 'Segoe UI'; font-size: 14px; }
            QMainWindow { background: #f5f7fb; }
            QGroupBox {
                border: 1px solid #d5deea;
                border-radius: 8px;
                margin-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #1f2937;
            }
            QPlainTextEdit {
                border: none;
                background: white;
                padding: 10px;
                font-size: 15px;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 9px 16px;
                min-width: 82px;
            }
            QPushButton:disabled { background: #94a3b8; }
            QPushButton:hover:!disabled { background: #1d4ed8; }
            QComboBox, QLineEdit {
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                padding: 7px 10px;
                background: white;
            }
            #status {
                color: #334155;
                padding: 8px 10px;
                background: #e8eef7;
                border-radius: 6px;
            }
            #historyTitle { color: #475569; font-weight: 600; }
            QPushButton[history="true"] {
                background: #e2e8f0;
                color: #0f172a;
                text-align: left;
                min-width: 112px;
            }
            QPushButton[history="true"]:hover { background: #cbd5e1; }
            """
        )

    def _wrap_group(self, title, widget):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 18, 10, 10)
        layout.addWidget(widget)
        return group

    def _load_devices(self):
        self.device_combo.clear()
        self.device_combo.addItem("默认麦克风", None)
        if sd is None:
            return
        try:
            for index, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    self.device_combo.addItem(f"{index}: {dev['name']}", index)
        except Exception as exc:
            self.status_label.setText(f"状态：读取麦克风失败：{exc}")

    def toggle_microphone(self):
        if self.worker is None:
            self.open_microphone()
        else:
            self.close_microphone()

    def open_microphone(self):
        self.thread = QThread()
        self.worker = RecognizerWorker(
            device_index=self.device_combo.currentData(),
            wake_word=self.current_wake_word(),
            stop_word=self.current_stop_word(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status_changed.connect(self.set_status)
        self.worker.mode_changed.connect(self._set_mode)
        self.worker.live_text_changed.connect(self.update_live_text)
        self.worker.command_started.connect(self.on_command_started)
        self.worker.command_finalized.connect(self.on_command_finalized)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.device_combo.setEnabled(False)
        self.mic_button.setEnabled(False)
        self.status_label.setText("状态：正在开启麦克风...")
        self.thread.start()

    def close_microphone(self):
        if self.worker is not None:
            self.worker.request_shutdown()
        self.mic_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态：正在关闭麦克风...")

    def start_command(self):
        if self.worker is None:
            self.status_label.setText("状态：请先点击开启")
            return
        self.worker.request_start(self.current_wake_word(), self.current_stop_word())

    def stop_command(self):
        if self.worker is not None:
            self.worker.request_stop()

    def on_wake_word_changed(self, text):
        if self.worker is not None:
            self.worker.set_wake_word(text)

    def on_stop_word_changed(self, text):
        if self.worker is not None:
            self.worker.set_stop_word(text)

    def on_command_started(self):
        self.archive_current_result()
        self.current_result = None
        self.current_result_archived = True
        self.live_text.clear()
        self.result_text.clear()

    def on_command_finalized(self, result):
        self.current_result = result
        self.current_result_archived = False
        self.show_result(result)

    def archive_current_result(self):
        if self.current_result is None or self.current_result_archived:
            return
        data = dict(self.current_result)
        self.history_items.append(data)
        index = len(self.history_items)
        button = QPushButton(f"第{index}次任务")
        button.setProperty("history", True)
        button.clicked.connect(lambda checked=False, item=data: self.show_result(item))
        self.history_list.insertWidget(self.history_list.count() - 1, button)
        self.current_result_archived = True

    def show_result(self, result):
        self.result_text.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))

    def update_live_text(self, text):
        self.live_text.setPlainText(text)
        self.live_text.moveCursor(QTextCursor.End)

    def set_status(self, message):
        self.status_label.setText(f"状态：{message}")

    def _set_mode(self, mode):
        self.mode = mode
        if mode == MIC_OFF:
            self.mic_button.setText("开启")
            self.mic_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.device_combo.setEnabled(True)
            if self.worker is None:
                self.status_label.setText("状态：未开启")
        elif mode == WAIT_WAKE:
            self.mic_button.setText("关闭")
            self.mic_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        elif mode == ACTIVE_COMMAND:
            self.mic_button.setText("关闭")
            self.mic_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def on_thread_finished(self):
        self.worker = None
        self.thread = None
        self._set_mode(MIC_OFF)
        self.status_label.setText("状态：未开启")

    def current_wake_word(self):
        return self.wake_word_edit.text().strip() or DEFAULT_WAKE_WORD

    def current_stop_word(self):
        return self.stop_word_edit.text().strip() or DEFAULT_STOP_WORD

    def show_error(self, message):
        QMessageBox.critical(self, "运行错误", message)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.request_shutdown()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(1500)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
