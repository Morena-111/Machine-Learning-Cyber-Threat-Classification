"""
Tkinter desktop client for the FastAPI cyber-threat classifier.

Design goals:
  - Light grey page background
  - Darker grey cards and header for clear visual separation
  - Vibrant emerald and purple accents
  - Large, bold typography
  - Colour-coded result cards (green / amber / red)
  - Keyboard shortcuts: Ctrl+Enter to classify, Ctrl+L to clear
"""

import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import requests


API_URL = "http://localhost:8000/predict"

# ============================================================================
# Palette: Light grey background, darker cards/header
# ============================================================================
BG_MAIN       = "#f1f5f9"   # Light grey page background
BG_HEADER     = "#e2e8f0"   # Darker grey header
BG_CARD       = "#e2e8f0"   # Darker grey cards
BG_CARD_INNER = "#f8fafc"   # Inner panel (slightly lighter for contrast)
BG_INPUT      = "#ffffff"   # White input box
BG_STATUS     = "#cbd5e1"   # Darkest grey status bar

TEXT_PRIMARY  = "#0f172a"   # Very dark slate for high contrast
TEXT_SECONDARY= "#334155"   # Medium slate
TEXT_MUTED    = "#64748b"   # Muted slate for hints

BORDER        = "#cbd5e1"   # Grey border
BORDER_FOCUS  = "#8b5cf6"   # Purple focus ring

# Buttons
BTN_PRIMARY_BG     = "#10b981"   # Bright emerald green
BTN_PRIMARY_HOVER  = "#059669"
BTN_PRIMARY_PRESS  = "#047857"
BTN_PRIMARY_TEXT   = "#ffffff"

BTN_SECONDARY_BG   = "#ffffff"
BTN_SECONDARY_HOVER= "#f1f5f9"
BTN_SECONDARY_PRESS= "#e2e8f0"
BTN_SECONDARY_TEXT = "#334155"

BTN_GHOST_BG       = "#cbd5e1"
BTN_GHOST_HOVER    = "#b4c0d0"
BTN_GHOST_PRESS    = "#94a3b8"
BTN_GHOST_TEXT     = "#334155"

# Status / result colours
SUCCESS       = "#10b981"
SUCCESS_BG    = "#ecfdf5"
SUCCESS_BORDER= "#10b981"
WARNING       = "#f59e0b"
WARNING_BG    = "#fffbeb"
WARNING_BORDER= "#f59e0b"
DANGER        = "#ef4444"
DANGER_BG     = "#fef2f2"
DANGER_BORDER = "#ef4444"

# Bar colours
BAR_TOP       = "#8b5cf6"
BAR_REST      = "#cbd5e1"


class ThreatClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyber Threat Classifier")
        self.geometry("1000x860")
        self.minsize(820, 720)
        self.configure(bg=BG_MAIN)
        self._bind_shortcuts()
        self._build_ui()

    def _bind_shortcuts(self):
        self.bind("<Control-Return>", lambda e: self.classify())
        self.bind("<Control-l>", lambda e: self.clear())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self, bg=BG_HEADER, height=120)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=BG_HEADER)
        header_inner.pack(fill="both", expand=True, padx=36, pady=22)

        title_row = tk.Frame(header_inner, bg=BG_HEADER)
        title_row.pack(anchor="w", fill="x")

        tk.Label(
            title_row,
            text="🛡  Cyber Threat Classifier",
            font=("Segoe UI", 26, "bold"),
            bg=BG_HEADER,
            fg=TEXT_PRIMARY,
        ).pack(side="left")

        tk.Label(
            title_row,
            text="   ·   BiLSTM + Attention   ·   20 classes",
            font=("Segoe UI", 12, "bold"),
            bg=BG_HEADER,
            fg=BTN_PRIMARY_BG,
        ).pack(side="left", padx=(4, 0))

        tk.Label(
            header_inner,
            text="Entity-level Named Entity Recognition for cyber threat intelligence",
            font=("Segoe UI", 12, "bold"),
            bg=BG_HEADER,
            fg=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(6, 0))

        # Divider
        tk.Frame(self, bg=BORDER, height=2).pack(fill="x")

        # ---- Scrollable content ----
        container = tk.Frame(self, bg=BG_MAIN)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=BG_MAIN, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            container, orient="vertical", command=canvas.yview,
            bg=BG_CARD, troughcolor=BG_MAIN, borderwidth=0,
            activebackground=BTN_PRIMARY_BG,
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(canvas, bg=BG_MAIN)
        canvas_window = canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)

        self.content.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_input_section(self.content)
        self._build_result_section(self.content)
        self._build_probability_section(self.content)
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Section ① — Input
    # ------------------------------------------------------------------
    def _build_input_section(self, parent):
        card = self._make_card(parent)
        card.pack(fill="x", padx=36, pady=(24, 16))

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="both", expand=True, padx=26, pady=22)

        header_row = tk.Frame(inner, bg=BG_CARD)
        header_row.pack(fill="x", anchor="w")

        tk.Label(
            header_row,
            text="①  ",
            font=("Segoe UI", 18, "bold"),
            bg=BG_CARD,
            fg=BTN_PRIMARY_BG,
        ).pack(side="left")

        tk.Label(
            header_row,
            text="INPUT THREAT INTELLIGENCE TEXT",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(side="left")

        tk.Label(
            inner,
            text="Paste CTI text (threat report, CVE reference, indicator, etc.)",
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(4, 14))

        # Input box wrapper (white)
        input_frame = tk.Frame(
            inner, bg=BG_INPUT,
            highlightbackground=BORDER, highlightthickness=2,
        )
        input_frame.pack(fill="x")

        self.text_input = tk.Text(
            input_frame,
            height=5,
            wrap="word",
            font=("Consolas", 11),
            bg=BG_INPUT,
            fg=TEXT_PRIMARY,
            insertbackground=BTN_PRIMARY_BG,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=12,
            selectbackground=BTN_PRIMARY_BG,
            selectforeground="#ffffff",
        )
        self.text_input.pack(fill="x")

        def _on_focus_in(event):
            input_frame.config(highlightbackground=BORDER_FOCUS)

        def _on_focus_out(event):
            input_frame.config(highlightbackground=BORDER)

        self.text_input.bind("<FocusIn>", _on_focus_in)
        self.text_input.bind("<FocusOut>", _on_focus_out)
        self.text_input.insert(
            "1.0",
            "The malware called Emotet is actively spreading",
        )

        # Buttons
        btn_row = tk.Frame(inner, bg=BG_CARD)
        btn_row.pack(fill="x", pady=(18, 0))

        self.classify_btn = self._make_button(
            btn_row, "🔍  Classify", self.classify, variant="primary"
        )
        self.classify_btn.pack(side="left")

        self._make_button(
            btn_row, "📂  Batch CSV", self.batch_predict, variant="secondary"
        ).pack(side="left", padx=(10, 0))

        self._make_button(
            btn_row, "✕  Clear", self.clear, variant="ghost"
        ).pack(side="right")

        tk.Label(
            inner,
            text="Tip:  Ctrl+Enter to classify  ·  Ctrl+L to clear",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(12, 0))

    # ------------------------------------------------------------------
    # Section ② — Result
    # ------------------------------------------------------------------
    def _build_result_section(self, parent):
        card = self._make_card(parent)
        card.pack(fill="x", padx=36, pady=(0, 16))

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="both", expand=True, padx=26, pady=22)

        header_row = tk.Frame(inner, bg=BG_CARD)
        header_row.pack(fill="x", anchor="w")

        tk.Label(
            header_row,
            text="②  ",
            font=("Segoe UI", 18, "bold"),
            bg=BG_CARD,
            fg=BTN_PRIMARY_BG,
        ).pack(side="left")

        tk.Label(
            header_row,
            text="PREDICTION RESULT",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(side="left")

        self.result_box = tk.Frame(
            inner, bg=BG_CARD_INNER,
            highlightbackground=BORDER, highlightthickness=2,
        )
        self.result_box.pack(fill="x", pady=(14, 0))

        result_inner = tk.Frame(self.result_box, bg=BG_CARD_INNER)
        result_inner.pack(fill="x", padx=20, pady=18)

        self.result_label = tk.Label(
            result_inner,
            text="Awaiting input…",
            font=("Segoe UI", 20, "bold"),
            bg=BG_CARD_INNER,
            fg=TEXT_SECONDARY,
            anchor="w",
        )
        self.result_label.pack(anchor="w")

        self.confidence_label = tk.Label(
            result_inner,
            text="",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD_INNER,
            fg=TEXT_PRIMARY,
            anchor="w",
        )
        self.confidence_label.pack(anchor="w", pady=(6, 12))

        self.confidence_canvas = tk.Canvas(
            result_inner, height=16, bg=BG_CARD_INNER,
            highlightthickness=0, bd=0,
        )
        self.confidence_canvas.pack(fill="x", pady=(0, 10))

        self.warning_label = tk.Label(
            result_inner,
            text="",
            font=("Segoe UI", 11, "bold"),
            bg=BG_CARD_INNER,
            fg=TEXT_MUTED,
            anchor="w",
        )
        self.warning_label.pack(anchor="w")

    # ------------------------------------------------------------------
    # Section ③ — Probabilities
    # ------------------------------------------------------------------
    def _build_probability_section(self, parent):
        card = self._make_card(parent)
        card.pack(fill="both", expand=True, padx=36, pady=(0, 24))

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="both", expand=True, padx=26, pady=22)

        header_row = tk.Frame(inner, bg=BG_CARD)
        header_row.pack(fill="x", anchor="w")

        tk.Label(
            header_row,
            text="③  ",
            font=("Segoe UI", 18, "bold"),
            bg=BG_CARD,
            fg=BTN_PRIMARY_BG,
        ).pack(side="left")

        tk.Label(
            header_row,
            text="CLASS PROBABILITIES",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(side="left")

        tk.Label(
            inner,
            text="Probability assigned to each of the 20 entity classes, sorted by confidence",
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(4, 14))

        self.prob_frame = tk.Frame(inner, bg=BG_CARD)
        self.prob_frame.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg=BG_STATUS, height=42)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_dot = tk.Label(
            status_bar,
            text="●",
            font=("Segoe UI", 16, "bold"),
            bg=BG_STATUS,
            fg=TEXT_MUTED,
            padx=24,
        )
        self.status_dot.pack(side="left", fill="y")

        self.status = tk.StringVar(value="API: not checked")
        tk.Label(
            status_bar,
            textvariable=self.status,
            font=("Segoe UI", 11, "bold"),
            bg=BG_STATUS,
            fg=TEXT_PRIMARY,
            anchor="w",
        ).pack(side="left", fill="both", expand=True)

        tk.Label(
            status_bar,
            text="ITRI625  ·  NER for CTI",
            font=("Segoe UI", 10, "bold"),
            bg=BG_STATUS,
            fg=TEXT_SECONDARY,
            padx=24,
        ).pack(side="right", fill="y")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_card(self, parent):
        return tk.Frame(
            parent,
            bg=BG_CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )

    def _make_button(self, parent, text, command, variant="primary"):
        if variant == "primary":
            bg, hover, press = BTN_PRIMARY_BG, BTN_PRIMARY_HOVER, BTN_PRIMARY_PRESS
            fg = BTN_PRIMARY_TEXT
            font = ("Segoe UI", 12, "bold")
            padx, pady = 28, 14
        elif variant == "secondary":
            bg, hover, press = BTN_SECONDARY_BG, BTN_SECONDARY_HOVER, BTN_SECONDARY_PRESS
            fg = BTN_SECONDARY_TEXT
            font = ("Segoe UI", 11, "bold")
            padx, pady = 24, 14
        else:
            bg, hover, press = BTN_GHOST_BG, BTN_GHOST_HOVER, BTN_GHOST_PRESS
            fg = BTN_GHOST_TEXT
            font = ("Segoe UI", 11)
            padx, pady = 22, 14

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=font,
            bg=bg,
            fg=fg,
            activebackground=press,
            activeforeground=fg,
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=padx,
            pady=pady,
            cursor="hand2",
            bd=1,
        )
        btn.config(highlightbackground=BORDER)

        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        btn.bind("<ButtonPress-1>", lambda e: btn.config(bg=press, relief="sunken"))
        btn.bind("<ButtonRelease-1>", lambda e: btn.config(bg=hover, relief="solid"))

        return btn

    def _draw_confidence_bar(self, confidence):
        self.confidence_canvas.delete("all")
        self.confidence_canvas.update_idletasks()
        w = self.confidence_canvas.winfo_width() or 800
        h = self.confidence_canvas.winfo_height() or 16

        if confidence >= 0.60:
            colour = SUCCESS
        elif confidence >= 0.30:
            colour = WARNING
        else:
            colour = DANGER

        fill_w = max(2, int(w * confidence))
        self.confidence_canvas.create_rectangle(0, 0, fill_w, h, fill=colour, outline="")

    def _draw_probability_bar(self, canvas, probability, is_top):
        canvas.delete("all")
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 22
        fill_w = max(2, int(w * probability))
        colour = BAR_TOP if is_top else BAR_REST
        canvas.create_rectangle(0, 0, fill_w, h, fill=colour, outline="")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def classify(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning(
                "Text required",
                "Please enter some threat intelligence text to classify.",
            )
            return

        self.status.set("Classifying…")
        self.status_dot.config(fg=WARNING)
        self.classify_btn.config(state="disabled", text="⏳  Classifying…")
        threading.Thread(
            target=self._request_and_display, args=(text,), daemon=True
        ).start()

    def _request(self, text):
        response = requests.post(API_URL, json={"text": text}, timeout=15)
        response.raise_for_status()
        return response.json()

    def _request_and_display(self, text):
        try:
            data = self._request(text)
            self.after(0, lambda: self.show_prediction(data))
        except requests.RequestException as error:
            self.after(0, lambda: self.connection_error(error))
        finally:
            self.after(
                0,
                lambda: self.classify_btn.config(
                    state="normal", text="🔍  Classify"
                ),
            )

    def show_prediction(self, data):
        predicted = data["predicted_class"]
        confidence = data["confidence"]

        if confidence >= 0.60:
            bg, fg, border = SUCCESS_BG, SUCCESS, SUCCESS_BORDER
            icon = "✓"
            message = f"{icon}  High confidence — prediction accepted"
        elif confidence >= 0.30:
            bg, fg, border = WARNING_BG, WARNING, WARNING_BORDER
            icon = "⚠"
            message = f"{icon}  Medium confidence — consider manual verification"
        else:
            bg, fg, border = DANGER_BG, DANGER, DANGER_BORDER
            icon = "⚠"
            message = f"{icon}  Low confidence — review manually"

        self.result_box.config(bg=bg, highlightbackground=border)
        for child in self.result_box.winfo_children():
            child.config(bg=bg)
            for grandchild in child.winfo_children():
                if isinstance(grandchild, (tk.Label, tk.Canvas)):
                    grandchild.config(bg=bg)

        self.result_label.config(
            text=f"Prediction:  {predicted}", bg=bg, fg=fg
        )
        self.confidence_label.config(
            text=f"Confidence:  {confidence:.1%}", bg=bg, fg=TEXT_PRIMARY
        )
        self.warning_label.config(text=message, bg=bg, fg=fg)

        self._draw_confidence_bar(confidence)

        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        sorted_probs = sorted(
            data["all_probabilities"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for i, (label, prob) in enumerate(sorted_probs):
            self._build_probability_row(i, label, prob)

        self.status.set(f"API: connected  ·  predicted '{predicted}'")
        self.status_dot.config(fg=SUCCESS)

    def _build_probability_row(self, index, label, probability):
        is_top = index == 0
        row = tk.Frame(self.prob_frame, bg=BG_CARD)
        row.pack(fill="x", pady=2)

        tk.Label(
            row,
            text=label,
            font=("Segoe UI", 10, "bold" if is_top else "normal"),
            bg=BG_CARD,
            fg=BAR_TOP if is_top else TEXT_PRIMARY,
            width=20,
            anchor="w",
        ).pack(side="left")

        bar = tk.Canvas(
            row, height=22, bg=BG_CARD_INNER, highlightthickness=0, bd=0
        )
        bar.pack(side="left", fill="x", expand=True, padx=(8, 12))

        def _redraw(event=None, canvas=bar, prob=probability, top=is_top):
            self._draw_probability_bar(canvas, prob, top)

        bar.bind("<Configure>", _redraw)

        tk.Label(
            row,
            text=f"{probability:.1%}",
            font=("Segoe UI", 10, "bold" if is_top else "normal"),
            bg=BG_CARD,
            fg=BAR_TOP if is_top else TEXT_SECONDARY,
            width=8,
            anchor="e",
        ).pack(side="right")

    def connection_error(self, error):
        self.status.set("API: unavailable")
        self.status_dot.config(fg=DANGER)
        messagebox.showerror(
            "Cannot reach API",
            "The FastAPI backend must be running.\n\n"
            "Start it with:\n\n"
            "    uvicorn api:app --reload --port 8000\n\n"
            f"Details: {error}",
        )

    def clear(self):
        self.text_input.delete("1.0", "end")
        self.result_box.config(bg=BG_CARD_INNER, highlightbackground=BORDER)
        for child in self.result_box.winfo_children():
            child.config(bg=BG_CARD_INNER)
            for grandchild in child.winfo_children():
                if isinstance(grandchild, (tk.Label, tk.Canvas)):
                    grandchild.config(bg=BG_CARD_INNER)
        self.result_label.config(
            text="Awaiting input…", fg=TEXT_SECONDARY, bg=BG_CARD_INNER
        )
        self.confidence_label.config(text="", bg=BG_CARD_INNER)
        self.warning_label.config(text="", bg=BG_CARD_INNER)
        self.confidence_canvas.delete("all")
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
        self.status.set("API: not checked")
        self.status_dot.config(fg=TEXT_MUTED)

    def batch_predict(self):
        source = filedialog.askopenfilename(
            title="Choose a CSV file with a 'text' column",
            filetypes=[("CSV files", "*.csv")],
        )
        if not source:
            return

        try:
            with open(source, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            if not rows or "text" not in rows[0]:
                raise ValueError("CSV must contain a 'text' column.")

            for i, row in enumerate(rows, start=1):
                self.status.set(f"Batch: {i}/{len(rows)}")
                self.status_dot.config(fg=WARNING)
                self.update_idletasks()
                response = self._request(row["text"])
                row["predicted_class"] = response["predicted_class"]
                row["confidence"] = f"{response['confidence']:.6f}"

            dest = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="batch_predictions.csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if dest:
                with open(dest, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                self.status.set(f"Saved {len(rows)} predictions")
                self.status_dot.config(fg=SUCCESS)
                messagebox.showinfo(
                    "Batch complete",
                    f"Saved {len(rows)} predictions to:\n{dest}",
                )
        except (OSError, ValueError, requests.RequestException) as error:
            self.status_dot.config(fg=DANGER)
            messagebox.showerror("Batch prediction failed", str(error))


if __name__ == "__main__":
    ThreatClassifierApp().mainloop()