"""Local desktop demo for manual text-image inference."""

from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.predict import predict_pair


ROOT = Path(__file__).resolve().parent
PREVIEW_SIZE = (460, 300)


class PredictionDemo:
    """Display frozen A/B predictions; run encoder work off the UI thread."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.root.title("Multimodal Fake News Demo")
        self.root.geometry("980x760")
        self.root.minsize(900, 700)

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Multimodal Fake News Detection",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Enter an unseen headline or short claim "
                "and choose its corresponding image."
            ),
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 14))

        input_frame = ttk.Frame(outer)
        input_frame.pack(fill="x")

        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)
        # Text input

        text_frame = ttk.LabelFrame(
            input_frame,
            text="News text",
            padding=10,
        )

        text_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        self.text_box = tk.Text(
            text_frame,
            height=11,
            wrap="word",
            font=("Segoe UI", 11),
        )

        self.text_box.pack(
            fill="both",
            expand=True,
        )

        ttk.Label(
            text_frame,
            text=(
                "Best input: a headline or short claim, "
                "matching the training setup."
            ),
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            pady=(8, 0),
        )
        # Image input

        image_frame = ttk.LabelFrame(
            input_frame,
            text="News image",
            padding=10,
        )

        image_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        self.preview = ttk.Label(
            image_frame,
            text="No image selected",
            anchor="center",
        )

        self.preview.pack(
            fill="both",
            expand=True,
        )

        self.image_name = ttk.Label(
            image_frame,
            text="",
        )

        self.image_name.pack(pady=(8, 5))

        ttk.Button(
            image_frame,
            text="Choose Image",
            command=self.choose_image,
        ).pack()
        # Controls

        controls = ttk.Frame(outer)

        controls.pack(
            fill="x",
            pady=16,
        )

        ttk.Label(
            controls,
            text="Checkpoint seed:",
        ).pack(side="left")

        self.seed_var = tk.StringVar(
            value="42",
        )

        seed_box = ttk.Combobox(
            controls,
            textvariable=self.seed_var,
            values=("42", "123", "2026"),
            width=8,
            state="readonly",
        )

        seed_box.pack(
            side="left",
            padx=(8, 18),
        )

        self.predict_button = ttk.Button(
            controls,
            text="Predict",
            command=self.start_prediction,
        )

        self.predict_button.pack(side="left")

        self.status_var = tk.StringVar(
            value="Ready",
        )

        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(
            side="left",
            padx=18,
        )
        # Results

        results = ttk.LabelFrame(
            outer,
            text="Results",
            padding=12,
        )

        results.pack(
            fill="both",
            expand=True,
        )

        results.columnconfigure(
            0,
            weight=1,
        )

        results.columnconfigure(
            1,
            weight=1,
        )

        self.similarity_var = tk.StringVar(
            value="CLIP cosine similarity: —",
        )

        ttk.Label(
            results,
            textvariable=self.similarity_var,
            font=("Segoe UI", 12, "bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 12),
        )
        # Baseline results

        baseline_frame = ttk.LabelFrame(
            results,
            text="Baseline — BERT + ResNet50",
            padding=12,
        )

        baseline_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        self.baseline_prediction = tk.StringVar(
            value="Prediction: —",
        )

        self.baseline_fake = tk.StringVar(
            value="Fake score: —",
        )

        self.baseline_real = tk.StringVar(
            value="Real score: —",
        )

        for variable in (
            self.baseline_prediction,
            self.baseline_fake,
            self.baseline_real,
        ):
            ttk.Label(
                baseline_frame,
                textvariable=variable,
                font=("Segoe UI", 11),
            ).pack(
                anchor="w",
                pady=4,
            )
        # Proposed results

        proposed_frame = ttk.LabelFrame(
            results,
            text=(
                "Proposed — "
                "BERT + ResNet50 + CLIP similarity"
            ),
            padding=12,
        )

        proposed_frame.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )

        self.proposed_prediction = tk.StringVar(
            value="Prediction: —",
        )

        self.proposed_fake = tk.StringVar(
            value="Fake score: —",
        )

        self.proposed_real = tk.StringVar(
            value="Real score: —",
        )

        for variable in (
            self.proposed_prediction,
            self.proposed_fake,
            self.proposed_real,
        ):
            ttk.Label(
                proposed_frame,
                textvariable=variable,
                font=("Segoe UI", 11),
            ).pack(
                anchor="w",
                pady=4,
            )

        ttk.Separator(results).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=14,
        )

        ttk.Label(
            results,
            text=(
                "Important: these are model softmax scores "
                "learned from Fakeddit, not independently "
                "verified truth probabilities."
            ),
            wraplength=850,
            font=("Segoe UI", 9),
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
        )

    def choose_image(self):
        selected = filedialog.askopenfilename(
            title="Choose a news image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.webp *.bmp",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not selected:
            return

        self.image_path = Path(selected)

        self.image_name.config(text=self.image_path.name)

        try:
            with Image.open(self.image_path) as image:
                preview = image.convert("RGB")

                preview.thumbnail(
                    PREVIEW_SIZE,
                    Image.Resampling.LANCZOS,
                )

                self.preview_photo = ImageTk.PhotoImage(
                    preview,
                )

        except Exception as exc:
            self.image_path = None
            self.preview_photo = None

            self.preview.config(
                image="",
                text="Could not preview image",
            )

            messagebox.showerror(
                "Image error",
                str(exc),
            )

            return

        self.preview.config(
            image=self.preview_photo,
            text="",
        )

    def start_prediction(self):
        text = self.text_box.get(
            "1.0",
            "end",
        ).strip()

        if not text:
            messagebox.showwarning(
                "Missing text",
                "Enter a headline or short claim first.",
            )
            return

        if self.image_path is None:
            messagebox.showwarning(
                "Missing image",
                "Choose an image first.",
            )
            return

        seed = int(
            self.seed_var.get(),
        )

        self.predict_button.config(state="disabled")

        self.status_var.set("Running BERT, ResNet50 and CLIP on CPU...")

        # Return results through Tk callbacks so widget updates stay on the UI thread.
        thread = threading.Thread(
            target=self._prediction_worker,
            args=(
                text,
                self.image_path,
                seed,
            ),
            daemon=True,
        )

        thread.start()

    def _prediction_worker(
        self,
        text: str,
        image_path: Path,
        seed: int,
    ):
        try:
            result = predict_pair(
                ROOT,
                text,
                image_path,
                seed,
            )

        except Exception as exc:
            self.root.after(
                0,
                self._prediction_failed,
                exc,
            )
            return

        self.root.after(
            0,
            self._show_result,
            result,
        )

    def _prediction_failed(
        self,
        exc: Exception,
    ):
        self.predict_button.config(state="normal")

        self.status_var.set("Prediction failed")

        messagebox.showerror(
            "Prediction error",
            str(exc),
        )

    def _show_result(
        self,
        result: dict,
    ):
        baseline = result["baseline"]
        proposed = result["proposed"]

        self.similarity_var.set(
            "CLIP cosine similarity: "
            f"{result['clip_similarity']:.6f}"
        )

        self.baseline_prediction.set(
            f"Prediction: "
            f"{baseline['prediction']}"
        )

        self.baseline_fake.set(
            "Fake score: "
            f"{baseline['probability_fake']:.2%}"
        )

        self.baseline_real.set(
            "Real score: "
            f"{baseline['probability_real']:.2%}"
        )

        self.proposed_prediction.set(
            f"Prediction: "
            f"{proposed['prediction']}"
        )

        self.proposed_fake.set(
            "Fake score: "
            f"{proposed['probability_fake']:.2%}"
        )

        self.proposed_real.set(
            "Real score: "
            f"{proposed['probability_real']:.2%}"
        )

        self.predict_button.config(state="normal")

        self.status_var.set("Prediction complete")


def main():
    root = tk.Tk()

    PredictionDemo(root)

    root.mainloop()


if __name__ == "__main__":
    main()