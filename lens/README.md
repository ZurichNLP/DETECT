🧭 DETECT: A Learnable Evaluation Metric for Text Simplification
DETECT is a new learned evaluation metric for text simplification, extending and improving upon LENS.
It provides fine-grained scoring across simplicity, meaning preservation, and fluency, along with a composite total score. DETECT builds on the LENS architecture and training methodology, while introducing new modeling and interpretability improvements.

🧩 Setup
Install from Source
bashgit clone https://github.com/your-org/DETECT.git
cd DETECT/detect


⚙️ Scoring within Python
The pretrained DETECT checkpoint is hosted on the Hugging Face Hub.
Example checkpoint: https://huggingface.co/your-org/detect
pythonfrom detect import DETECT

# Initialize model
# rescale=True rescales z-scores to [0, 100] for interpretability.
detect = DETECT("your-org/detect:best.ckpt", rescale=True)

complex = [
    "They are culturally akin to the coastal peoples of Papua New Guinea."
]

simple = [
    "They are culturally similar to the people of Papua New Guinea."
]

references = [[
    "They are culturally similar to the coastal peoples of Papua New Guinea.",
    "They are similar to the Papua New Guinea people living on the coast."
]]

scores = detect.score(complex, simple, references, batch_size=8, devices=[0])
print(scores)
# [{'simplicity': 78.6, 'meaning_preservation': 80.1, 'fluency': 77.3, 'total': 78.3}]
Output Dimensions
Each output dictionary contains four fields:
KeyDescriptionsimplicityHow much simpler the output is compared to the inputmeaning_preservationHow well the original meaning is preservedfluencyGrammaticality and readability of the simplificationtotalWeighted combination of all three (see below)

🧮 Total Score Calculation
By default, the total score is computed as:
pythondef total(s, m, f):
    # Weighted combination
    # 0.4 * simplicity + 0.4 * meaning_preservation + 0.2 * fluency
    # If any component < 25, total = min(s, m, f)
    if min(s, m, f) < 25:
        return min(s, m, f)
    return 0.4 * s + 0.4 * m + 0.2 * f
This reflects DETECT's focus on overall quality, while penalizing outputs that fail any individual dimension.

📖 API Reference
Initialization
pythondetect = DETECT(
    lens_path: str,       # e.g. "your-org/detect:checkpoint.ckpt"
    rescale: bool = True  # rescales z-scores to [0,100]
)
Scoring
pythondetect.score(
    complex: list[str],
    simple: list[str],
    references: list[list[str]],
    batch_size: int = 8,
    devices: list[int] = [0],
    features: Optional[list] = None,
    show_progress: bool = False,
)
Returns:
python[
  {
    "simplicity": float,
    "meaning_preservation": float,
    "fluency": float,
    "total": float
  },
  ...
]

🧠 Relationship to LENS
DETECT builds directly on LENS, reusing its COMET-inspired architecture and training framework.
However, DETECT introduces:

Separate interpretable dimensions (simplicity, meaning preservation, fluency)
A calibrated [0–100] output scale
A weighted composite scoring rule (total(s,m,f))
Model fine-tuning on new simplification evaluation data

If you are looking for the original single-score metric, please refer to the LENS repository.

🧾 Citation
If you use DETECT, please cite our paper:
bibtex@inproceedings{yourname2025detect,
  title = "{DETECT}: Dimensionally Explainable Text Evaluation for Simplification",
  author = "Your Name and Collaborators",
  booktitle = "Proceedings of Conference",
  year = "2025",
  publisher = "Association",
}

📝 License
[Specify your license here, e.g., MIT, Apache 2.0]
🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
📧 Contact
For questions or issues, please contact [your-email@example.com] or open an issue on GitHub.