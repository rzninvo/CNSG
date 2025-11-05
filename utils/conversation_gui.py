from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton
from PySide6.QtCore import Qt, Slot, QTimer, QCoreApplication
from PySide6.QtGui import QFont, QTextCursor, QTextBlockFormat, QTextCharFormat
from html import escape
import queue
import threading
import os

# Dependencies:
# pip install PySide6 SpeechRecognition gTTS playsound
# conda install -c conda-forge pyaudio alsa-plugins jack speex
# On Linux for TTS: sudo apt-get install espeak

# Optional Speech Recognition support
try:
    import speech_recognition as sr  # type: ignore
    SR_AVAILABLE = True
except Exception:
    sr = None
    print("Speech recognition support not available. Install SpeechRecognition and PyAudio for voice input.")
    SR_AVAILABLE = False


# Optional Text-to-Speech (TTS) support
try:
    from gtts import gTTS
    import playsound
    TTS_AVAILABLE = True
except Exception as e:
    gTTS = None
    playsound = None
    print(f"Text-to-Speech (TTS) support not available (pip install gTTS playsound). Error: {e}")
    TTS_AVAILABLE = False



class ModernGui(QWidget):
    """
    Modern chat GUI.
    """

    def __init__(self, input_q: queue.Queue, output_q: queue.Queue, window_width: int, window_height: int):
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.mic_q = queue.Queue()
        self.tts_q = queue.Queue()      # Queue for messages to be spoken
        self.tts_enabled = False        # TTS state
        self.loading_block_number = -1
        self.is_listening = False       # State flag for mic interrupt

        self.init_ui(window_width, window_height)
        self.apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_output_queue)
        self.timer.start(100)

        if TTS_AVAILABLE:
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def init_ui(self, window_width: int, window_height: int):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        header_label = QLabel("Navigation Assistant Chat")
        header_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(300)
        main_layout.addWidget(self.log_area, 1)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Where would you like to go?")
        self.input_entry.returnPressed.connect(self.submit_input)
        input_layout.addWidget(self.input_entry, 1)

        self.speaker_button = QPushButton()
        self.speaker_button.setText("\U0001F507") # Muted icon
        self.speaker_button.setObjectName("speakerButton")
        self.speaker_button.setCursor(Qt.PointingHandCursor)
        self.speaker_button.setFixedSize(40, 40)
        self.speaker_button.setToolTip(
            "Toggle Text-to-Speech (TTS) (TTS is OFF)"
            if TTS_AVAILABLE
            else "TTS not available (install gTTS, playsound)"
        )
        self.speaker_button.clicked.connect(self._on_speaker_toggled)
        if not TTS_AVAILABLE:
            self.speaker_button.setEnabled(False)
        input_layout.addWidget(self.speaker_button)

        self.mic_button = QPushButton()
        self.mic_button.setText("\U0001F399") # Studio mic icon
        self.mic_button.setObjectName("micButton")
        self.mic_button.setCursor(Qt.PointingHandCursor)
        self.mic_button.setFixedSize(40, 40)
        self.mic_button.setToolTip(
            "Press and speak for voice input"
            if SR_AVAILABLE
            else "Speech recognition not available (install SpeechRecognition/pyaudio)"
        )
        self.mic_button.clicked.connect(self._on_mic_clicked)
        if not SR_AVAILABLE:
            self.mic_button.setEnabled(False)
        input_layout.addWidget(self.mic_button)

        self.submit_button = QPushButton("Send")
        self.submit_button.setCursor(Qt.PointingHandCursor)
        self.submit_button.clicked.connect(self.submit_input)
        input_layout.addWidget(self.submit_button)

        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)
        self.resize(window_width, window_height)

    def apply_style(self):
        self.setStyleSheet("""
            ModernGui {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            QLabel { color: #FFFFFF; }
            QTextEdit {
                background-color: #2A2A2A;
                border-radius: 10px;
                padding: 10px;
                font-size: 11pt;
            }
            QLineEdit {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 11pt;
                color: white;
            }
            QLineEdit:focus { border-color: #0A84FF; }
            QPushButton {
                background-color: #0A84FF;
                border-radius: 20px;
                padding: 10px 22px;
                color: white;
                font-weight: bold;
                font-size: 11pt;
            }
                         
            QPushButton:hover { background-color: #3AA0FF; }
            
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
                           
            #micButton {
                border-radius: 20px; 
                padding: 0px;        
                font-size: 16pt;     
                background-color: #333;
                border: 1px solid #555;
                padding-top: 6px;
            }
            #micButton:hover {
                background-color: #555;
            }
            #micButton:pressed {
                background-color: #0A84FF; 
            }
            
            #speakerButton {
                border-radius: 20px;
                padding: 0px;
                font-size: 16pt;     
                background-color: #333;
                border: 1px solid #555;
                padding-top: 6px;
            }
            #speakerButton:hover { background-color: #555; }
        """)

    # ---------- Helpers ----------
    def _append_message(self, name_html: str, body_html: str, align: Qt.AlignmentFlag, color_css: str, is_loading: bool = False) -> int:
        """Appends a single chat message."""
        cursor = self.log_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_area.setTextCursor(cursor)

        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(align)
        block_fmt.setTopMargin(0)
        block_fmt.setBottomMargin(3)
        cursor.insertBlock(block_fmt)
        
        block_number = cursor.blockNumber()

        html = (
            f'<span style="display:inline-block; padding:6px 6px; '
            f'border-radius:18px; font-size:11pt; color:{color_css}; '
            f'max-width:70%; background:transparent; word-wrap:break-word;">'
            f'{name_html}{"" if is_loading else "<br>"}{body_html}'
            f'</span>'
        )
        cursor.insertHtml(html)

        self.scroll_to_bottom()
        return block_number

    def _show_loading_indicator(self):
        """Shows the '...' indicator."""
        loading_html = ('<b style="font-size:9pt;">Assistant...</b>')
        
        self.loading_block_number = self._append_message(
            name_html='',
            body_html=loading_html,
            align=Qt.AlignLeft,
            color_css="#E5E5EA",
            is_loading=True
        )

    def _remove_loading_indicator(self):
        """Removes the loading indicator's text block."""
        if self.loading_block_number == -1:
            return

        document = self.log_area.document()
        block = document.findBlockByNumber(self.loading_block_number)

        if block.isValid():
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.BlockUnderCursor)
            
            if not block.next().isValid() or block.next().text().strip() == "":
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            
            cursor.removeSelectedText()

        self.loading_block_number = -1 # Reset
        try:
            QCoreApplication.processEvents()
        except Exception:
            pass
            
    def _reset_mic_ui(self):
        """Resets the microphone UI to its default state."""
        self.is_listening = False
        self._remove_loading_indicator()
        try:
            self.mic_button.setEnabled(True)
            self.mic_button.setText("\U0001F399")
        except Exception:
            pass
        self.submit_button.setEnabled(True)
        self.input_entry.setEnabled(True)
        if TTS_AVAILABLE:
            self.speaker_button.setEnabled(True)

    # ---------- Slots ----------
    @Slot()
    def submit_input(self):
        """Triggered by Send or Enter key."""
        text = self.input_entry.text().strip()
        if not text:
            return

        if text.lower() == "clear":
            self._remove_loading_indicator()
            self.log_area.clear()
            try:
                while True:
                    self.output_q.get_nowait()
            except queue.Empty:
                pass
            self.input_entry.clear()
            return

        self.input_q.put(text)
        escaped = escape(text).replace("\n", "<br>")
        self._append_message('<b style="font-size:9pt;">You</b>', escaped, Qt.AlignRight, "white")
        self.input_entry.clear()
        self._show_loading_indicator()

    @Slot()
    def _on_mic_clicked(self):
        """Toggles background listening on or off (interrupts)."""
        if not SR_AVAILABLE:
            self._append_message(
                '<b style="font-size:9pt;">System</b>',
                escape("Speech recognition not available. Install SpeechRecognition and PyAudio."),
                Qt.AlignLeft,
                "#E5E5EA",
            )
            return

        if self.is_listening:
            # --- INTERRUPT/CANCEL CURRENT RECORDING ---
            print("[GUI] Mic interrupt clicked.")
            self._reset_mic_ui()
        
        else:
            # --- START NEW RECORDING ---
            self.is_listening = True
            # self.mic_button.setEnabled(False)
            try:
                self.mic_button.setText("⏺️")
            except Exception:
                pass
            self.submit_button.setEnabled(False)
            self.input_entry.setEnabled(False)
            self.speaker_button.setEnabled(False) 

            self._show_loading_indicator()

            t = threading.Thread(target=self._listen_worker, daemon=True)
            t.start()
        
    @Slot()
    def _on_speaker_toggled(self):
        """Toggles Text-to-Speech on/off."""
        self.tts_enabled = not self.tts_enabled
        if self.tts_enabled:
            self.speaker_button.setText("\U0001F50A") # Speaker icon
            self.speaker_button.setToolTip("TTS is ON")
        else:
            self.speaker_button.setText("\U0001F507") # Muted icon
            self.speaker_button.setToolTip("TTS is OFF")

    # ---------- Workers and Handlers ----------

    def _listen_worker(self) -> None:
        """Background worker to capture audio and perform SR."""
        if not SR_AVAILABLE:
            self.mic_q.put((False, "Speech recognition unavailable."))
            return

        print("[MicWorker] Thread started.")
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.5 
        
        try:
            print("[MicWorker] Initializing Microphone...")
            with sr.Microphone() as source:
                print("[MicWorker] Adjusting for ambient noise (0.8s)...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                
                print("[MicWorker] Listening... (max 12s)")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
                print("[MicWorker] Audio captured. Recognizing...")

            try:
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"[MicWorker] Recognition success: {text}")
                
                if self.is_listening: # Check if not canceled
                    self.mic_q.put((True, text))
                else:
                    print("[MicWorker] Canceled by user. Discarding result.")

            except Exception as e:
                print(f"[MicWorker] Recognition error: {e}")
                msg = f"Recognition error: {e}"
                if hasattr(sr, "UnknownValueError") and isinstance(e, sr.UnknownValueError):
                    msg = "I didn't understand. Can you repeat?"
                elif hasattr(sr, "RequestError") and isinstance(e, sr.RequestError):
                    msg = f"Service unavailable: {e}"
                
                if self.is_listening: # Check if not canceled
                    self.mic_q.put((False, msg))
                else:
                    print("[MicWorker] Canceled by user. Discarding error.")
                return

        except Exception as e:
            print(f"[MicWorker] CRITICAL ERROR (Microphone?): {e}")
            if self.is_listening: # Check if not canceled
                self.mic_q.put((False, f"Microphone error: {e}"))


    def _on_recognition_result(self, success: bool, text: str) -> None:
        """Handles the recognition result on the GUI thread."""
        self._reset_mic_ui() # Reset UI state first
        
        try:
            QCoreApplication.processEvents()
        except Exception:
            pass

        if success:
            self.input_entry.setText(text)
            QTimer.singleShot(50, self.submit_input)
        else:
            self._append_message('<b style="font-size:9pt;">System</b>', escape(text).replace("\n", "<br>"), Qt.AlignLeft, "#E5E5EA")

    def _tts_worker(self) -> None:
        """
        Background worker that waits for messages on self.tts_q
        and plays them using gTTS and playsound.
        """
        if not TTS_AVAILABLE:
            return
            
        print("[TTSWorker] Thread started.")
        tts_filename = "tts_output.mp3" # Temporary file
        
        while True:
            try:
                text_to_speak = self.tts_q.get()
                
                if text_to_speak is None: # Exit
                    break
                
                print(f"[TTSWorker] Generating audio for: {text_to_speak}")
                tts = gTTS(text=text_to_speak, lang='en')
                tts.save(tts_filename)
                
                print("[TTSWorker] Playing...")
                playsound.playsound(tts_filename)
                print("[TTSWorker] Playback finished.")
                
            except Exception as e:
                print(f"[TTSWorker] Error: {e}")
            finally:
                try:
                    if os.path.exists(tts_filename):
                        os.remove(tts_filename)
                except Exception:
                    pass # Not critical

    def check_output_queue(self):
        """Polls the queues for messages and results."""
        # 1. Check for assistant messages
        try:
            while True:
                msg = self.output_q.get_nowait()
                
                self._remove_loading_indicator() 
                
                escaped = escape(msg).replace("\n", "<br>")
                self._append_message('<b style="font-size:9pt;">Assistant</b>', escaped, Qt.AlignLeft, "#E5E5EA")

                if self.tts_enabled and TTS_AVAILABLE:
                    self.tts_q.put(msg) 

        except queue.Empty:
            pass

        # 2. Check for mic results
        try:
            while True:
                success, text = self.mic_q.get_nowait()
                self._on_recognition_result(success, text)
        except queue.Empty:
            pass

    def scroll_to_bottom(self):
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())


def create_gui(input_q: queue.Queue, output_q: queue.Queue, window_width: int, window_height: int) -> QWidget:
    return ModernGui(input_q, output_q, window_width, window_height)