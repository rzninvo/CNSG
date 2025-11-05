from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton
from PySide6.QtCore import Qt, Slot, QTimer, QCoreApplication
from PySide6.QtGui import QFont, QTextCursor, QTextBlockFormat, QTextCharFormat
from html import escape
import queue


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
        
        # Variabile per tracciare il blocco (paragrafo) del loading
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
        # Forza un aggiornamento visivo
        QCoreApplication.processEvents()


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

    def check_output_queue(self):
        """Poll assistant messages and append them."""
        try:
            while True:
                msg = self.output_q.get_nowait()
                
                self._remove_loading_indicator() 
                
                escaped = escape(msg).replace("\n", "<br>")
                self._append_message('<b style="font-size:9pt;">Assistant</b>', escaped, Qt.AlignLeft, "#E5E5EA")
        except queue.Empty:
            pass

    def scroll_to_bottom(self):
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())


def create_gui(input_q: queue.Queue, output_q: queue.Queue, window_width: int, window_height: int) -> QWidget:
    return ModernGui(input_q, output_q, window_width, window_height)