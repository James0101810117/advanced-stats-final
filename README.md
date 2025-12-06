# Delivery Delay Dashboard & Late Delivery Risk Models

This repository contains a Streamlit application and supporting documents for analyzing delivery delays and predicting **`Late_delivery_risk`** using the **DataCoSupplyChain** dataset.

The project combines:
- An **interactive Market-level delay dashboard**
- Several **classification models** (Logistic Regression, Random Forest, and optionally XGBoost) to predict whether an order will be delayed.

---

## 📁 Repository Structure

```text
Project/
├─ App/
│  ├─ app.py               # Main Streamlit app (dashboard + ML models)
│  └─ module1.py           # (Optional) Extra helper functions / modules
│
├─ Research_Paper/
│  ├─ Paper1.pdf           # Key academic paper 1 (e.g., ESG / SCM)
│  └─ Paper2.pdf           # Key academic paper 2
│
├─ report/
│  └─ report.pdf           # Final written report (PDF)
│
├─ presentation/
│  └─ project.pdf          # Project presentation slides (PDF)
│
├─ data/
│  └─ DataCoSupplyChainDataset_sample_20000.csv  # Dataset (not always included in repo)
│
├─ requirements.txt        # Python dependencies
└─ README.md               # This file
Note: The data file might not be publicly committed due to size or confidentiality.
Place the CSV inside the data/ folder on your local machine.

📊 1. Delivery Delay Dashboard (Market-level)
The upper part of the Streamlit app provides an interactive dashboard:

Delay definition:
Late_delivery_risk

1 = delayed delivery

0 = on-time delivery

Key features:

Market-level aggregation:

Average delay rate per Market

Number of orders per Market

Sidebar filters:

Select which markets to display

Filter markets by minimum number of orders

KPIs:

Total number of valid orders

Overall average delay rate

Number of orders in selected markets

Visualization:

Bar chart of delay rate by Market (Plotly if available; fallback to Streamlit bar chart)

Detail table:

Market, order volume, and delay rate (%) for each Market

This part helps answer:

“Which markets have the highest delay rate, and are these results reliable given the sample size?”

🤖 2. Late_delivery_risk – Predictive Models (80/20 split)
The lower part of the app trains and evaluates several models to predict whether an order will be late.

Models
Logistic Regression (L2 regularization)

Random Forest Classifier

XGBoost Classifier (optional, if xgboost is installed)

All models are trained on rows with a valid Late_delivery_risk value using:

Train/test split: 80% training, 20% testing

Stratified split to respect the class distribution

Data preprocessing
The app uses a ColumnTransformer + Pipeline:

Numeric features:

Missing values imputed with mean

Standardized with StandardScaler

Categorical features:

Missing values imputed with most frequent

Encoded with OneHotEncoder (handle_unknown="ignore")

Leakage & ID-like feature removal
Before training, several columns are removed to avoid data leakage and meaningless identifiers, e.g.:

Leakage examples:

Late_delivery_risk (target itself)

Delivery Status

Days for shipping (real)

Scheduled delivery date

Shipping date (DateOrders) (or similar)

ID-like columns:

Order Id, Order Item Id, Customer Id, Customer Email, etc.

The idea is to keep only features that would be available before shipping and that are generalizable.

Metrics
For each model, the app reports:

Accuracy

F1-score

AUC (ROC AUC)

Results are shown in a table (for the test set).

Feature Importance (Random Forest)
For the Random Forest model:

The app extracts feature importances

Builds a table of the top 5 most important features

Displays a horizontal bar chart of these top features

This part helps answer:

“How well can we predict late deliveries, and which variables are most important for the prediction?”

🛠 3. Installation
3.1. Clone or download the repository
You can either:

Clone with Git:

bash
複製程式碼
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
or

Download the ZIP from GitHub and extract it, then go into the project folder.

3.2. Create and activate a virtual environment (recommended)
bash
複製程式碼
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
3.3. Install dependencies
From the project root (where requirements.txt is located):

bash
複製程式碼
pip install -r requirements.txt
The main libraries used include:

streamlit

pandas, numpy

scikit-learn

matplotlib

plotly (optional, for interactive charts)

xgboost (optional, for the XGBoost model)

▶️ 4. Running the Streamlit App
From the project root, run:

bash
複製程式碼
streamlit run App/app.py
This will open the app in your browser (or give you a local URL such as http://localhost:8501).

📂 5. Dataset Setup
Download the DataCoSupplyChain dataset (or the sample file used in this project).

Save it into the data/ folder, e.g.:

text
複製程式碼
data/DataCoSupplyChainDataset_sample_20000.csv
In the Streamlit app (left sidebar), there is a “CSV file path” input box.

You can either:

Paste the full absolute path, e.g.:

text
複製程式碼
C:\Users\james\...\data\DataCoSupplyChainDataset_sample_20000.csv
or

Use a relative path from the project root, e.g.:

text
複製程式碼
data/DataCoSupplyChainDataset_sample_20000.csv
If the file cannot be loaded, the app will show a clear error message.

📑 6. Project Documents
report/report.pdf
Final written report, explaining:

Business problem

Data preparation & feature engineering

Model selection & evaluation

Managerial insights and recommendations

presentation/project.pdf
Slide deck used for the in-class / final presentation.

Research_Paper/
Contains the key academic references used in the project and report.

👥 7. Contributors
Team member: Wei-Chun LAN (James),Hsin-Wei HUANG

⚠️ 8. Disclaimer
This project is built for academic purposes only.
The dataset and results are used for learning and demonstration, not for real operational decisions.
