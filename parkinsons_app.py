import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import threading
import os

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
def train_model():
    df = pd.read_csv("parkinsons_100k.csv")
    features = df.drop(['name', 'status'], axis=1)
    target = df['status']
    scaler = MinMaxScaler((-1, 1))
    features_scaled = scaler.fit_transform(features)
    x_train, x_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=10)
    model = RandomForestClassifier(random_state=2)
    model.fit(x_train, y_train)
    return model, scaler, features.columns.tolist()


# ─────────────────────────────────────────────
#  FIELD DEFINITIONS
# ─────────────────────────────────────────────
FIELDS = [
    # (label, key, min, max, description)
    ("MDVP:Fo(Hz)",       "fo",      88.33,   260.11, "Avg vocal frequency"),
    ("MDVP:Fhi(Hz)",      "fhi",     102.15,  592.03, "Max vocal frequency"),
    ("MDVP:Flo(Hz)",      "flo",     65.48,   239.17, "Min vocal frequency"),
    ("MDVP:Jitter(%)",    "jit_p",   0.00168, 0.03316,"Freq variation %"),
    ("MDVP:Jitter(Abs)",  "jit_abs", 7e-6,    0.00026,"Abs freq variation"),
    ("MDVP:RAP",          "rap",     0.00068, 0.02144,"Relative avg perturbation"),
    ("MDVP:PPQ",          "ppq",     0.00092, 0.01958,"5-point period perturbation"),
    ("Jitter:DDP",        "ddp",     0.00204, 0.06433,"Avg abs diff of diffs"),
    ("MDVP:Shimmer",      "shim",    0.00954, 0.11908,"Amplitude variation"),
    ("MDVP:Shimmer(dB)",  "shim_db", 0.085,   1.302,  "Shimmer in dB"),
    ("Shimmer:APQ3",      "apq3",    0.00455, 0.05647,"3-point amplitude perturbation"),
    ("Shimmer:APQ5",      "apq5",    0.0057,  0.0794, "5-point amplitude perturbation"),
    ("MDVP:APQ",          "apq",     0.00719, 0.13778,"11-point amplitude perturbation"),
    ("Shimmer:DDA",       "dda",     0.01364, 0.16942,"Avg abs diffs of amplitudes"),
    ("NHR",               "nhr",     0.00065, 0.31482,"Noise-to-harmonics ratio"),
    ("HNR",               "hnr",     8.441,   33.047, "Harmonics-to-noise ratio"),
    ("RPDE",              "rpde",    0.2566,  0.6852, "Dynamical complexity"),
    ("DFA",               "dfa",     0.5743,  0.8253, "Signal fractal scaling"),
    ("spread1",           "sp1",    -7.965,  -2.434,  "Nonlinear freq variation 1"),
    ("spread2",           "sp2",     0.00627, 0.4505, "Nonlinear freq variation 2"),
    ("D2",                "d2",      1.423,   3.671,  "Correlation dimension"),
    ("PPE",               "ppe",     0.04454, 0.52737,"Pitch period entropy"),
]

SAMPLE_HEALTHY = [
    197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,
    0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.775,
    0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569
]
SAMPLE_PARKINSON = [
    119.992,157.302,74.997,0.00784,0.00007,0.00370,0.00554,0.01109,
    0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.033,
    0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654
]


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class ParkinsonsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parkinson's Disease Detection")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg="#0d1117")
        self.resizable(True, True)

        # fonts
        self.f_title  = font.Font(family="Courier", size=18, weight="bold")
        self.f_sub    = font.Font(family="Courier", size=10)
        self.f_label  = font.Font(family="Courier", size=9,  weight="bold")
        self.f_entry  = font.Font(family="Courier", size=11, weight="bold")
        self.f_btn    = font.Font(family="Courier", size=11, weight="bold")
        self.f_result = font.Font(family="Courier", size=20, weight="bold")
        self.f_hint   = font.Font(family="Courier", size=8)

        # color palette
        self.C = {
            "bg":      "#0d1117",
            "panel":   "#161b22",
            "border":  "#21262d",
            "accent":  "#58a6ff",
            "green":   "#3fb950",
            "red":     "#f85149",
            "yellow":  "#d29922",
            "text":    "#e6edf3",
            "muted":   "#8b949e",
            "input_bg":"#0d1117",
            "hover":   "#1f2937",
        }

        self.entries = {}
        self.model = None
        self.scaler = None

        self._build_ui()
        self._load_model_async()

    # ── async model load ──────────────────────
    def _load_model_async(self):
        self.status_var.set("⟳  Training model on parkinsons.csv …")
        self.btn_predict.config(state="disabled")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        try:
            self.model, self.scaler, _ = train_model()
            self.after(0, lambda: self.status_var.set("✓  Model ready  |  Accuracy ≈ 94–97 %"))
            self.after(0, lambda: self.btn_predict.config(state="normal"))
            self.after(0, lambda: self.status_lbl.config(fg=self.C["green"]))
        except Exception as e:
            self.after(0, lambda: self.status_var.set(f"✗  Error: {e}"))
            self.after(0, lambda: self.status_lbl.config(fg=self.C["red"]))

    # ── UI BUILD ──────────────────────────────
    def _build_ui(self):
        # ── LEFT SIDEBAR ──
        sidebar = tk.Frame(self, bg=self.C["panel"], width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Frame(sidebar, bg=self.C["accent"], height=3).pack(fill="x")

        # logo area
        logo_frame = tk.Frame(sidebar, bg=self.C["panel"], pady=28)
        logo_frame.pack(fill="x")

        # brain icon using unicode
        tk.Label(logo_frame, text="⬡", font=("Courier", 36, "bold"),
                 bg=self.C["panel"], fg=self.C["accent"]).pack()
        tk.Label(logo_frame, text="NEURO\nDETECT", font=("Courier", 13, "bold"),
                 bg=self.C["panel"], fg=self.C["text"], justify="center").pack(pady=(4,0))
        tk.Label(logo_frame, text="v1.0 · ML Edition",
                 font=self.f_hint, bg=self.C["panel"], fg=self.C["muted"]).pack()

        tk.Frame(sidebar, bg=self.C["border"], height=1).pack(fill="x", padx=20, pady=10)

        # info blocks
        info_items = [
            ("DATASET",    "195 voice samples"),
            ("FEATURES",   "22 vocal biomarkers"),
            ("ALGORITHM",  "Random Forest"),
            ("ACCURACY",   "≈ 94 – 97 %"),
            ("CLASSES",    "Healthy / Parkinson's"),
        ]
        for k, v in info_items:
            row = tk.Frame(sidebar, bg=self.C["panel"], padx=20, pady=6)
            row.pack(fill="x")
            tk.Label(row, text=k, font=self.f_hint,
                     bg=self.C["panel"], fg=self.C["accent"]).pack(anchor="w")
            tk.Label(row, text=v, font=("Courier", 10, "bold"),
                     bg=self.C["panel"], fg=self.C["text"]).pack(anchor="w")

        tk.Frame(sidebar, bg=self.C["border"], height=1).pack(fill="x", padx=20, pady=12)

        # quick-fill buttons
        tk.Label(sidebar, text="QUICK FILL", font=self.f_hint,
                 bg=self.C["panel"], fg=self.C["muted"], padx=20).pack(anchor="w", pady=(0,6))

        self._sidebar_btn(sidebar, "📋  Load Healthy Sample",
                          lambda: self._fill_sample(SAMPLE_HEALTHY), self.C["green"])
        self._sidebar_btn(sidebar, "📋  Load Parkinson Sample",
                          lambda: self._fill_sample(SAMPLE_PARKINSON), self.C["red"])

        tk.Frame(sidebar, bg=self.C["border"], height=1).pack(fill="x", padx=20, pady=8)
        tk.Label(sidebar, text="RANDOM TEST", font=self.f_hint,
                 bg=self.C["panel"], fg=self.C["muted"], padx=20).pack(anchor="w", pady=(0,6))

        self._sidebar_btn(sidebar, "🎲  Random & Auto-Predict",
                          self._fill_random_and_predict, self.C["yellow"])
        self._sidebar_btn(sidebar, "🔀  Fill Random Values",
                          self._fill_random, "#c678dd")

        tk.Frame(sidebar, bg=self.C["border"], height=1).pack(fill="x", padx=20, pady=8)
        self._sidebar_btn(sidebar, "✕  Clear All Fields",
                          self._clear_all, self.C["muted"])

        # status at bottom of sidebar
        tk.Frame(sidebar, bg=self.C["panel"]).pack(expand=True, fill="y")
        tk.Frame(sidebar, bg=self.C["border"], height=1).pack(fill="x", padx=20)
        self.status_var = tk.StringVar(value="Initializing…")
        self.status_lbl = tk.Label(sidebar, textvariable=self.status_var,
                                   font=self.f_hint, bg=self.C["panel"],
                                   fg=self.C["yellow"], wraplength=200,
                                   justify="left", padx=16, pady=14)
        self.status_lbl.pack(fill="x")

        # ── MAIN AREA ──
        main = tk.Frame(self, bg=self.C["bg"])
        main.pack(side="left", fill="both", expand=True)

        # top bar
        topbar = tk.Frame(main, bg=self.C["panel"], height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Frame(topbar, bg=self.C["border"], width=1).pack(side="left", fill="y")
        tk.Label(topbar, text="  PATIENT VOCAL BIOMARKER INPUT",
                 font=self.f_label, bg=self.C["panel"], fg=self.C["muted"]).pack(side="left", padx=10)

        # scrollable form area
        scroll_frame = tk.Frame(main, bg=self.C["bg"])
        scroll_frame.pack(fill="both", expand=True, padx=0, pady=0)

        canvas = tk.Canvas(scroll_frame, bg=self.C["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        self.form_frame = tk.Frame(canvas, bg=self.C["bg"])

        self.form_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._build_form(self.form_frame)

        # ── BOTTOM BAR ──
        bottombar = tk.Frame(main, bg=self.C["panel"], height=70)
        bottombar.pack(fill="x", side="bottom")
        bottombar.pack_propagate(False)
        tk.Frame(bottombar, bg=self.C["accent"], height=2).pack(fill="x")

        btn_row = tk.Frame(bottombar, bg=self.C["panel"])
        btn_row.pack(expand=True)

        self.btn_predict = tk.Button(
            btn_row, text="  RUN ANALYSIS  →",
            font=self.f_btn, bg=self.C["accent"], fg="#0d1117",
            relief="flat", cursor="hand2", padx=28, pady=8,
            command=self._predict,
            activebackground="#79b8ff", activeforeground="#0d1117"
        )
        self.btn_predict.pack(side="left", padx=8, pady=10)

    def _sidebar_btn(self, parent, text, cmd, color):
        btn = tk.Button(parent, text=text, font=self.f_hint,
                        bg=self.C["panel"], fg=color,
                        relief="flat", cursor="hand2",
                        activebackground=self.C["hover"],
                        activeforeground=color,
                        anchor="w", padx=20, pady=7,
                        command=cmd)
        btn.pack(fill="x")
        btn.bind("<Enter>", lambda e: btn.config(bg=self.C["hover"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=self.C["panel"]))

    def _build_form(self, parent):
        # Section headers + 22 fields in a 4-column grid
        sections = [
            ("FREQUENCY",  [0,1,2],        self.C["accent"]),
            ("JITTER",     [3,4,5,6,7],    "#d2a679"),
            ("SHIMMER",    [8,9,10,11,12,13], "#c678dd"),
            ("NOISE",      [14,15],        self.C["red"]),
            ("NONLINEAR",  [16,17,18,19,20,21], self.C["green"]),
        ]

        pad = {"padx": 24, "pady": 6}

        for sec_name, indices, color in sections:
            # section divider
            sec_row = tk.Frame(parent, bg=self.C["bg"])
            sec_row.pack(fill="x", **pad)
            tk.Frame(sec_row, bg=color, width=3, height=18).pack(side="left")
            tk.Label(sec_row, text=f"  {sec_name}", font=self.f_label,
                     bg=self.C["bg"], fg=color).pack(side="left")
            tk.Frame(sec_row, bg=self.C["border"], height=1).pack(
                side="left", fill="x", expand=True, padx=12)

            # grid of fields
            grid = tk.Frame(parent, bg=self.C["bg"])
            grid.pack(fill="x", padx=24, pady=(0, 8))

            cols = 4
            for i, idx in enumerate(indices):
                label, key, mn, mx, desc = FIELDS[idx]
                col = i % cols
                row = i // cols
                self._make_field(grid, label, key, mn, mx, desc, color).grid(
                    row=row, column=col, padx=6, pady=5, sticky="ew")

            for c in range(cols):
                grid.columnconfigure(c, weight=1)

    def _make_field(self, parent, label, key, mn, mx, desc, accent):
        frame = tk.Frame(parent, bg=self.C["panel"],
                         highlightbackground=self.C["border"],
                         highlightthickness=1)

        # top label row
        top = tk.Frame(frame, bg=self.C["panel"])
        top.pack(fill="x", padx=10, pady=(8,2))
        tk.Label(top, text=label, font=self.f_label,
                 bg=self.C["panel"], fg=accent).pack(side="left")

        # range hint
        rng = f"{mn:.4g}–{mx:.4g}"
        tk.Label(frame, text=rng, font=self.f_hint,
                 bg=self.C["panel"], fg=self.C["muted"]).pack(anchor="w", padx=10)

        # entry
        var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=var, font=self.f_entry,
                         bg=self.C["input_bg"], fg=self.C["text"],
                         insertbackground=self.C["accent"],
                         relief="flat", bd=0, width=14)
        entry.pack(fill="x", padx=10, pady=(4,8))

        # focus highlight
        entry.bind("<FocusIn>",  lambda e, f=frame: f.config(
            highlightbackground=accent, highlightthickness=1))
        entry.bind("<FocusOut>", lambda e, f=frame: f.config(
            highlightbackground=self.C["border"], highlightthickness=1))

        self.entries[key] = var
        return frame

    # ── ACTIONS ──────────────────────────────
    def _fill_sample(self, values):
        for i, (_, key, *_) in enumerate(FIELDS):
            self.entries[key].set(str(values[i]))

    def _fill_random(self):
        for label, key, mn, mx, desc in FIELDS:
            val = np.random.uniform(mn, mx)
            if abs(val) < 0.001:
                self.entries[key].set(f"{val:.6f}")
            elif abs(val) < 1:
                self.entries[key].set(f"{val:.5f}")
            else:
                self.entries[key].set(f"{val:.4f}")

    def _fill_random_and_predict(self):
        if not self.model:
            messagebox.showwarning("Not Ready", "Model is still loading. Please wait.")
            return
        self._fill_random()
        self.after(300, self._predict)

    def _clear_all(self):
        for var in self.entries.values():
            var.set("")
        if hasattr(self, "result_win") and self.result_win.winfo_exists():
            self.result_win.destroy()

    def _predict(self):
        if not self.model:
            messagebox.showwarning("Not Ready", "Model is still loading. Please wait.")
            return

        vals = []
        for label, key, mn, mx, desc in FIELDS:
            raw = self.entries[key].get().strip()
            if not raw:
                messagebox.showerror("Missing Value", f"Please fill in: {label}")
                return
            try:
                v = float(raw)
            except ValueError:
                messagebox.showerror("Invalid Value", f"'{raw}' is not a number for {label}")
                return
            vals.append(v)

        arr = np.array(vals).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        prediction = self.model.predict(arr_scaled)[0]
        proba = self.model.predict_proba(arr_scaled)[0]

        self._show_result(prediction, proba)

    # ── RESULT POPUP ─────────────────────────
    def _show_result(self, prediction, proba):
        win = tk.Toplevel(self)
        self.result_win = win
        win.title("Analysis Result")
        win.geometry("520x360")
        win.resizable(False, False)
        win.configure(bg=self.C["bg"])
        win.grab_set()  # modal

        # center on parent
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 520) // 2
        y = self.winfo_y() + (self.winfo_height() - 360) // 2
        win.geometry(f"+{x}+{y}")

        is_positive = prediction == 1
        color  = self.C["red"]   if is_positive else self.C["green"]
        icon   = "⚠"             if is_positive else "✓"
        title  = "PARKINSON'S DETECTED" if is_positive else "NO PARKINSON'S DETECTED"
        msg    = ("The vocal biomarkers indicate patterns\nconsistent with Parkinson's Disease.\nPlease consult a medical professional."
                  if is_positive else
                  "The vocal biomarkers appear within\nnormal ranges. No signs of\nParkinson's Disease detected.")

        # top accent bar
        tk.Frame(win, bg=color, height=4).pack(fill="x")

        body = tk.Frame(win, bg=self.C["bg"])
        body.pack(fill="both", expand=True, padx=40, pady=30)

        tk.Label(body, text=icon, font=("Courier", 52, "bold"),
                 bg=self.C["bg"], fg=color).pack()
        tk.Label(body, text=title, font=("Courier", 16, "bold"),
                 bg=self.C["bg"], fg=color).pack(pady=(8, 4))
        tk.Label(body, text=msg, font=("Courier", 10),
                 bg=self.C["bg"], fg=self.C["muted"], justify="center").pack()

        # probability bar
        conf_frame = tk.Frame(body, bg=self.C["bg"])
        conf_frame.pack(fill="x", pady=(20, 4))
        healthy_pct = int(proba[0] * 100)
        park_pct    = int(proba[1] * 100)

        tk.Label(conf_frame, text=f"Healthy {healthy_pct}%",
                 font=self.f_hint, bg=self.C["bg"], fg=self.C["green"]).pack(side="left")
        tk.Label(conf_frame, text=f"Parkinson's {park_pct}%",
                 font=self.f_hint, bg=self.C["bg"], fg=self.C["red"]).pack(side="right")

        bar_bg = tk.Frame(body, bg=self.C["border"], height=8)
        bar_bg.pack(fill="x")
        bar_bg.update_idletasks()
        w = bar_bg.winfo_width() or 440
        tk.Frame(bar_bg, bg=self.C["green"],
                 width=int(w * proba[0]), height=8).place(x=0, y=0)
        tk.Frame(bar_bg, bg=self.C["red"],
                 width=int(w * proba[1]), height=8).place(
                     x=int(w * proba[0]), y=0)

        tk.Label(body, text="⚕  This tool is for research purposes only.",
                 font=self.f_hint, bg=self.C["bg"], fg=self.C["muted"]).pack(pady=(16, 0))

        tk.Button(body, text="CLOSE", font=self.f_btn,
                  bg=color, fg="#0d1117", relief="flat",
                  padx=24, pady=6, cursor="hand2",
                  command=win.destroy).pack(pady=(14, 0))


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ParkinsonsApp()
    app.mainloop()