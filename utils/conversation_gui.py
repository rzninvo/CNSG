from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton
from PySide6.QtCore import Qt, Slot, QTimer, QCoreApplication
from PySide6.QtGui import QFont, QTextCursor, QTextBlockFormat, QTextCharFormat
from html import escape
import queue
import threading

# pip install PySide6
# pip install SpeechRecognition
# conda install -c conda-forge pyaudio
# conda install -c conda-forge alsa-plugins jack speex

# Optional speech recognition support
try:
    import speech_recognition as sr  # type: ignore
    SR_AVAILABLE = True
except Exception:
    sr = None
    print("Speech recognition support not available. Install SpeechRecognition and PyAudio for voice input.")
    SR_AVAILABLE = False


class ModernGui(QWidget):
    """
    Modern chat GUI styled like Apple iMessage (transparent bubbles).
    - User messages: right aligned
    - Assistant messages: left aligned
    - Robust alignment via QTextBlockFormat (no HTML align)
    """

    def __init__(self, input_q: queue.Queue, output_q: queue.Queue, window_width: int, window_height: int):
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.mic_q = queue.Queue()  
        self.loading_block_number = -1 


        self.init_ui(window_width, window_height)
        self.apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_output_queue)
        self.timer.start(100)

    # --- (init_ui e apply_style sono rimaste le stesse per brevità) ---
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

        # Microphone button for voice input
        self.mic_button = QPushButton()
        self.mic_button.setText("\U0001F3A4")
        self.mic_button.setObjectName("micButton")
        self.mic_button.setCursor(Qt.PointingHandCursor)
        self.mic_button.setFixedSize(40, 40)
        self.mic_button.setToolTip(
            "Press and speak to enter a voice query"
            if SR_AVAILABLE
            else "Speech recognition not available (install SpeechRecognition/pyaudio)"
        )
        self.mic_button.clicked.connect(self._on_mic_clicked)
        # Disable if speech recognition support is missing
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
                border-radius: 20px; /* Metà di 40px */
                padding: 0px;        /* Remove problemcatic padding */
                font-size: 16pt;     /* Makes the emoji bigger */
                background-color: #333;
                border: 1px solid #555;
            }
            #micButton:hover {
                background-color: #555;
            }
            #micButton:pressed {
                background-color: #0A84FF; 
            }

        """)
    # -------------------------------------------------------------

    # ---------- Helpers ----------
    def _append_message(self, name_html: str, body_html: str, align: Qt.AlignmentFlag, color_css: str, is_loading: bool = False) -> int:
        """Append a single chat message with robust paragraph alignment and returns the block number."""
        cursor = self.log_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_area.setTextCursor(cursor)

        # New paragraph with explicit alignment (this is the key fix)
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(align)
        block_fmt.setTopMargin(0)     # compact vertical spacing
        block_fmt.setBottomMargin(3)
        cursor.insertBlock(block_fmt)
        
        # Salviamo il numero del blocco prima di inserire l'HTML
        block_number = cursor.blockNumber()

        # Transparent 'bubble' that sizes to content
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
        """Mostra l'indicatore di caricamento '...' e traccia il blocco."""
        # CSS per l'animazione dei puntini (necessario per l'animazione)
        loading_html = (
            '<b style="font-size:9pt;">Assistant...</b>'
        )
        
        # Inseriamo l'indicatore e salviamo il numero di blocco
        self.loading_block_number = self._append_message(
            name_html='',
            body_html=loading_html,
            align=Qt.AlignLeft,
            color_css="#E5E5EA",
            is_loading=True
        )

    def _remove_loading_indicator(self):
        """Rimuove il blocco di testo (paragrafo) dell'indicatore di caricamento."""
        if self.loading_block_number == -1:
            return

        document = self.log_area.document()
        # Otteniamo il blocco in base al numero tracciato
        block = document.findBlockByNumber(self.loading_block_number)

        if block.isValid():
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.BlockUnderCursor)
            
            # Un piccolo trucco per rimuovere anche il potenziale spazio vuoto
            # se il blocco successivo non esiste o è vuoto (per pulire i margini)
            if not block.next().isValid() or block.next().text().strip() == "":
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            
            cursor.removeSelectedText()

        self.loading_block_number = -1 # Reset
        # Forza un aggiornamento visivo: assicurati che il GUI thread rinfreschi
        try:
            QCoreApplication.processEvents()
        except Exception:
            # If something goes wrong, silently continue
            pass


    # ---------- Slots ----------
    @Slot()
    def submit_input(self):
        """Triggered when user presses Enter or clicks Send."""
        text = self.input_entry.text().strip()
        if not text:
            return

        # Special command: clear the chat window and pending assistant outputs
        if text.lower() == "clear":
            # Remove any loading indicator if present
            self._remove_loading_indicator()

            # Clear visible chat area
            self.log_area.clear()

            # Drain any pending assistant messages in the output queue to avoid
            # them reappearing after a clear (they are considered stale)
            try:
                while True:
                    self.output_q.get_nowait()
            except queue.Empty:
                pass

            # Clear the input box and return (do not forward "clear" to logic thread)
            self.input_entry.clear()
            return

        # Normal input: forward to logic thread and show user message + loading
        self.input_q.put(text)

        escaped = escape(text).replace("\n", "<br>")
        self._append_message('<b style="font-size:9pt;">You</b>', escaped, Qt.AlignRight, "white")
        self.input_entry.clear()
        
        self._show_loading_indicator()

    @Slot()
    def _on_mic_clicked(self):
        """Start background listening if SR is available."""
        if not SR_AVAILABLE:
            self._append_message(
                '<b style="font-size:9pt;">System</b>',
                escape("Speech recognition not available. Install SpeechRecognition and PyAudio."),
                Qt.AlignLeft,
                "#E5E5EA",
            )
            return

        # Update UI state
        self.mic_button.setEnabled(False)
        try:
            self.mic_button.setText("⏺️")
        except Exception:
            print("Error setting mic button text.")
            pass
        self.submit_button.setEnabled(False)
        self.input_entry.setEnabled(False)

        # Show small loading indicator in chat
        self._show_loading_indicator()

        # Launch recognition in background
        t = threading.Thread(target=self._listen_worker, daemon=True)
        t.start()
        
    def _listen_worker(self) -> None:
        """Background worker that captures audio and performs speech recognition."""
        if not SR_AVAILABLE:
            # ERRORE: Metti il risultato nella coda
            self.mic_q.put((False, "Speech recognition unavailable."))
            return

        print("[MicWorker] Thread started.")
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.5
        try:
            print("[MicWorker] Initializing Microphone...")
            with sr.Microphone() as source:
                # ... (adjust_for_ambient_noise)
                print("[MicWorker] Adjusting for ambient noise (0.8s)...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                
                print("[MicWorker] Listening... (max 12s)")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
                print("[MicWorker] Audio captured. Recognizing...")

            try:
                text = recognizer.recognize_google(audio, language="it-IT")
                print(f"[MicWorker] Recognition success: {text}")
                # SUCCESSO: Metti il risultato nella coda
                self.mic_q.put((True, text))

            except Exception as e:
                print(f"[MicWorker] Recognition error: {e}")
                # ERRORE: Metti il risultato nella coda
                msg = f"Errore di riconoscimento: {e}"
                if hasattr(sr, "UnknownValueError") and isinstance(e, sr.UnknownValueError):
                    msg = "Non ho capito. Puoi ripetere?"
                elif hasattr(sr, "RequestError") and isinstance(e, sr.RequestError):
                    msg = f"Servizio non disponibile: {e}"
                
                self.mic_q.put((False, msg))
                return

        except Exception as e:
            print(f"[MicWorker] CRITICAL ERROR (Microphone?): {e}")
            # ERRORE: Metti il risultato nella coda
            self.mic_q.put((False, f"Errore microfono: {e}"))


    def _on_recognition_result(self, success: bool, text: str) -> None:
        """Handle the recognition result on the GUI thread."""
        # remove listening indicator and restore buttons
        self._remove_loading_indicator()
        try:
            self.mic_button.setEnabled(True)
            # Reset visual pressed state and icon/text
            try:
                self.mic_button.setDown(False)
            except Exception:
                pass
            self.mic_button.setText("🎤")
        except Exception:
            print("Error restoring mic button text.")
            pass
        self.submit_button.setEnabled(True)
        self.input_entry.setEnabled(True)

        # Ensure the UI refreshes immediately so the button and loading indicator
        # don't visually remain stuck while other work happens.
        try:
            QCoreApplication.processEvents()
        except Exception:
            print("Error processing events after recognition.")
            pass

        if success:
            # Put text into the input and submit as if the user typed it
            self.input_entry.setText(text)
            # Small delay so UI updates before submitting
            QTimer.singleShot(50, self.submit_input)
            # self.submit_input()
        else:
            # Show friendly system message in chat
            self._append_message('<b style="font-size:9pt;">System</b>', escape(text).replace("\n", "<br>"), Qt.AlignLeft, "#E5E5EA")

    def check_output_queue(self):
        """Poll assistant messages and append them."""
        # 1. Controlla i messaggi dell'assistente (come prima)
        try:
            while True:
                msg = self.output_q.get_nowait()
                
                self._remove_loading_indicator() 
                
                escaped = escape(msg).replace("\n", "<br>")
                self._append_message('<b style="font-size:9pt;">Assistant</b>', escaped, Qt.AlignLeft, "#E5E5EA")
        except queue.Empty:
            pass

        # 2. AGGIUNGI QUESTO: Controlla i risultati del microfono
        try:
            while True:
                # Prendi il risultato dalla coda
                success, text = self.mic_q.get_nowait()
                
                # Chiama _on_recognition_result (ora siamo nel thread GUI, è sicuro)
                self._on_recognition_result(success, text)
        except queue.Empty:
            pass

    def scroll_to_bottom(self):
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())


def create_gui(input_q: queue.Queue, output_q: queue.Queue, window_width: int, window_height: int) -> QWidget:
    return ModernGui(input_q, output_q, window_width, window_height)