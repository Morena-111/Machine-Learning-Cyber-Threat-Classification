"""Tkinter desktop client for the local FastAPI classifier."""
import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import requests

API_URL = "http://localhost:8000/predict"
BG, CARD, ACCENT = "#102a43", "#ffffff", "#0b7285"


class ThreatClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyber Threat Intelligence Classifier")
        self.geometry("780x650")
        self.configure(bg=BG)
        self._build()

    def _build(self):
        card = tk.Frame(self, bg=CARD, padx=24, pady=20)
        card.pack(fill="both", expand=True, padx=22, pady=22)
        tk.Label(card, text="Cyber Threat Intelligence Classifier", font=("Segoe UI", 18, "bold"), bg=CARD, fg=BG).pack(anchor="w")
        tk.Label(card, text="Paste a CTI report to predict its primary threat category.", bg=CARD, fg="#486581").pack(anchor="w", pady=(3, 12))
        self.text = tk.Text(card, height=10, wrap="word", font=("Segoe UI", 11), relief="solid", borderwidth=1)
        self.text.pack(fill="x")
        controls = tk.Frame(card, bg=CARD)
        controls.pack(fill="x", pady=12)
        tk.Button(controls, text="Classify", command=self.classify, bg=ACCENT, fg="white", font=("Segoe UI", 10, "bold"), padx=20, relief="flat").pack(side="left")
        tk.Button(controls, text="Batch CSV prediction", command=self.batch_predict, bg="#d9e2ec", fg=BG, padx=15, relief="flat").pack(side="left", padx=10)
        self.result = tk.Label(card, text="Awaiting input", font=("Segoe UI", 13, "bold"), bg=CARD, fg=BG)
        self.result.pack(anchor="w", pady=(8, 3))
        self.warning = tk.Label(card, text="", font=("Segoe UI", 10), bg=CARD, fg="#c92a2a")
        self.warning.pack(anchor="w")
        self.bars = tk.Frame(card, bg=CARD)
        self.bars.pack(fill="both", expand=True, pady=8)
        self.status = tk.StringVar(value="API: not checked")
        tk.Label(self, textvariable=self.status, bg="#081f33", fg="white", anchor="w", padx=12).pack(fill="x", side="bottom")

    def classify(self):
        text = self.text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Text required", "Enter threat intelligence text before classifying.")
            return
        self.status.set("Classifying…")
        threading.Thread(target=self._request_and_display, args=(text,), daemon=True).start()

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

    def show_prediction(self, data):
        confidence = data["confidence"]
        self.result.config(text=f"Prediction: {data['predicted_class']} ({confidence:.1%})")
        self.warning.config(text="⚠ Low-confidence result: review manually." if confidence < 0.60 else "")
        for widget in self.bars.winfo_children():
            widget.destroy()
        for label, probability in sorted(data["all_probabilities"].items(), key=lambda item: item[1], reverse=True):
            row = tk.Frame(self.bars, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, width=18, anchor="w", bg=CARD).pack(side="left")
            progress = ttk.Progressbar(row, maximum=1, value=probability)
            progress.pack(side="left", fill="x", expand=True, padx=8)
            tk.Label(row, text=f"{probability:.1%}", width=7, anchor="e", bg=CARD).pack(side="right")
        self.status.set("API: connected")

    def connection_error(self, error):
        self.status.set("API: unavailable")
        messagebox.showerror("Cannot reach API", f"Start the backend with `uvicorn api:app --reload`.\n\n{error}")

    def batch_predict(self):
        source = filedialog.askopenfilename(title="Choose CSV", filetypes=[("CSV files", "*.csv")])
        if not source:
            return
        try:
            with open(source, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "text" not in rows[0]:
                raise ValueError("CSV must contain a 'text' column.")
            for row in rows:
                response = self._request(row["text"])
                row["predicted_class"] = response["predicted_class"]
                row["confidence"] = f"{response['confidence']:.6f}"
            destination = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="batch_predictions.csv")
            if destination:
                with open(destination, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
                self.status.set(f"Saved {len(rows)} predictions")
        except (OSError, ValueError, requests.RequestException) as error:
            messagebox.showerror("Batch prediction failed", str(error))


if __name__ == "__main__":
    ThreatClassifierApp().mainloop()
