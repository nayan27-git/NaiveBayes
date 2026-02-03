# Play Tennis - Naive Bayes Implementation

A modular machine learning pipeline built with Scikit-Learn to predict whether to play tennis based on weather conditions. This project demonstrates a shift from experimental Jupyter notebooks to a production-ready **Object-Oriented Programming (OOP)** structure.

##  Features

* **Custom Data Transformer:** A robust cleaning class that handles missing values for categorical weather features.
* **Encapsulated Pipeline:** Uses `ColumnTransformer` and Scikit-Learn `Pipeline` to bundle preprocessing and the `MultinomialNB` model together.
* **Configuration-Driven:** Model parameters and data paths are managed via a `config.yaml` file for easy experimentation.
* **Detailed Evaluation:** Automatically calculates Accuracy, Precision, Recall, F1 Score, and Log Loss.

---

##  Project Structure

```text
play_tennis_project/
├── src/
│   ├── __init__.py
│   └── model.py            # Contains DataTransformer and ModelManager classes
├── Data/
│   └── play_tennis.csv     # Dataset
├── config.yaml             # Control panel for features and parameters
├── run_model.py            # Entry point to train and evaluate
├── requirements.txt        # Project dependencies
└── .gitignore              # Files excluded from version control

```

---

##  Setup & Installation

1. **Clone the repository:**
```bash
git clone <your-repo-link>
cd play_tennis_project

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

##  Usage

1. **Configure the project:**
Edit `config.yaml` to change hyperparameters (like `alpha` for Naive Bayes) or to update the list of features.
2. **Run the model:**
Execute the driver script from the root directory:
```bash
python run_model.py

```



---

##  Metrics Tracked

The model evaluates performance using:

* **Accuracy:** Overall correctness.
* **F1 Score:** Harmonic mean of precision and recall for the "Yes" label.
* **Log Loss:** Measures the performance of probability predictions.
* **Confusion Matrix:** To visualize true vs. predicted classifications.

