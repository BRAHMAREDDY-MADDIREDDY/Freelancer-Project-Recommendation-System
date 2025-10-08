## 🎯 Freelancer Project Recommendation System

### Project Overview

This project presents a full-stack Freelancer Project Recommendation System that provides personalized job suggestions for freelancers. It utilizes advanced machine learning techniques, including Content-Based Filtering using BERT embeddings, Neural Collaborative Filtering (NCF), and a Hybrid model that combines both methods. The system recommends freelance projects based on a user’s skills and interaction history. Key features include secure user login and registration, an interactive job browsing interface, feedback collection, and the ability to export recommended jobs for later use.

## 📦 Dataset

This project uses two datasets to power the recommendation system:

- A [freelance job dataset](https://www.kaggle.com/datasets/isaacoresanya/freelancer) containing job titles, descriptions, categories, and associated tags or skills.
- A [contracts dataset](https://www.kaggle.com/datasets/asaniczka/freelance-contracts-dataset-1-3-million-entries) with historical records of freelance jobs, including freelancer IDs, job details, and payment information.

Both datasets were obtained from publicly available sources on Kaggle.

## 🤖 Machine Learning Models Used

**1. Content-Based Filtering**  
This model uses a fine-tuned BERT architecture to generate embeddings for job descriptions and freelancer profiles. Jobs are ranked based on their semantic similarity to the freelancer's profile.

**2. Neural Collaborative Filtering (NCF)**  
This model learns interaction patterns between freelancers and jobs by analyzing past job completions. It captures user preferences to recommend relevant projects.

**3. Hybrid Model**  
This model integrates both content-based and collaborative filtering approaches by combining BERT and NCF embeddings through multi-modal fusion and attention mechanisms. This fusion improves the accuracy and relevance of job recommendations by leveraging both semantic and interaction-based signals.

## 🧩 MVP Core Features Overview

- **User Authentication:** Secure signup and login with hashed passwords to protect user credentials.  
- **Home Screen:** Personalized welcome message with top job recommendations; allows toggling between content-based, collaborative filtering, and hybrid models.  
- **Freelancer Profile Input & Job Recommendation:** Supports manual and file-upload profile inputs; generates job recommendations using BERT-based content filtering.  
- **Interaction-Based Recommendation:** Lets users simulate past job interactions; uses Neural Collaborative Filtering to predict relevant jobs and rank recommendations.  
- **Hybrid Recommendation System:** Combines content-based and collaborative filtering results for improved recommendations with comparative model views.  
- **Inference Interface for Hybrid Model:** Accepts freelancer ID and project details to provide binary fit predictions along with confidence scores.  


## 🧩 Additional Features Overview

- **Clickable Job Cards:** Interactive job listings with detailed descriptions for better user engagement.  
- **Skill Gap Analyzer:** Recommends new skills based on comparison between user skills and job required skills.  
- **Tag-Based Job Filtering:** Allows filtering jobs by selected skills or tags to narrow down recommendations.  
- **Export Recommendations:** Enables exporting top recommended jobs to CSV or JSON formats for offline use.  
- **Watchlist:** Lets users save and manage favorite jobs for easy future access.  
- **Feedback System:** Collects likes and dislikes on job recommendations to improve future model retraining.  
- **Profile Skill Extractor:** Uses NLP techniques to automatically extract skills from user-uploaded resumes or descriptions.  

## 🛠 Project Setup and Installation Instructions

1. Clone the repository:

```bash
git clone https://github.com/DSA_1_Freelancer_Project_recommnedation_Sytstem.git
cd DSA_1_Freelancer_Project_recommnedation_Sytstem
```

- After Cloning you will get below Project structure

## 📁 Project Structure

```bash
DSA_1_Freelancer_Project_recommnedation_Sytstem/
├── backend/
│   ├── config.py                           # Configuration settings (e.g., environment variables, constants)
│   ├── content_prerecommendations.py       # Logic for content-based filtering recommendations
│   ├── main.py                             # Entry point for backend API (FastAPI)
│   ├── models.py                           # SQLAlchemy models defining DB schema
│   ├── ncf_recommendations.py              # Neural Collaborative Filtering (NCF) logic
│   ├── profile_extractor.py                # User profile and skill extraction logic
│   ├── recommendations.py                  # Recommendation engine orchestration and integration
│   └── skills_list.txt                     # Predefined list of skills used in processing
│
├── data/
│   ├── my_embedding_model/                 # Directory containing custom embedding model files
│   │   ├── config.json                     # Model configuration (used by tokenizer)
│   │   ├── model.h5                        # Trained Keras/TensorFlow model
│   │   ├── special_tokens_map.json         # Tokenizer-specific special tokens mapping
│   │   ├── tf_model.h5                     # Alternative TensorFlow model file
│   │   ├── tokenizer_config.json           # Configuration for tokenizer behavior
│   │   └── vocab.txt                       # Tokenizer vocabulary list
│   ├── preprocessed/
│   │   ├── ncf_dataset_preprocessed.pkl                              # Preprocessed dataset for NCF model
│   │   ├── ncf_dataset_preprocessed_job_ids_with_descriptions.pkl    # Preprocessed job ID and description data
│   │   └── ncf_dataset_preprocessed_tfidf_cols.pkl                   # TF-IDF feature columns for hybrid model
│   ├── bert_dataset_raw.csv                # Raw job dataset for BERT training
│   ├── item_encoder.pkl                    # Pickled item (job) encoder
│   ├── job_vecs.pkl                        # Pickled job embeddings
│   ├── ncf_dataset.csv                     # Raw interaction dataset for NCF model
│   ├── ncf_model.h5                        # Trained NCF model
│   ├── test_dataset_raw.csv                # Raw test dataset
│   ├── train_dataset_raw.csv               # Raw training dataset
│   └── user_encoder.pkl                    # Pickled user encoder
│
├── database/
│   ├── crud.py                             # DB operations: create, read, update, delete 
│   ├── db.py                               # DB connection/session management
│   └── tables.sql                          # SQL schema definitions
│
├── frontend/
│   ├── __init__.py                         # Marks frontend as a Python module
│   ├── freelancer_hub_log.png              # Application logo or banner
│   └── streamlit_app.py                    # Streamlit-based frontend UI
│
├── pre_process_ncf_data.py                # Script to preprocess data for NCF model
├── precompute_embeddings.py               # Script to precompute job/user embeddings
├── README.md                              # Project documentation
├── .gitignore                             # Git ignore file to exclude cache, logs, models, etc.
└── requirements.txt                       # List of Python dependencies for the project

```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL database:

- Create a new PostgreSQL database for the project.

- Configure database connection details in a .env file at the root of the project. Example .env file content:

```bash
DATABASE_URL=postgresql://username:password@localhost:5432/freelancer_db
```

- Run the `tables.sql` script to create the necessary tables (`users`, `watchlist`, `feedback`, and `recommendation`):

```bash
psql -d freelancer_db -f tables.sql
```

## ⚙️ .gitignore Description

To keep the repository clean and secure, the following are excluded from version control via `.gitignore`:

- **`data/` folder:**  
  This folder contains large datasets, pre-trained models, and intermediate files that can be very large and are not necessary to include in the repository. Users can download required datasets separately following the instructions.

- **`.env` file:**  
  This file stores sensitive environment variables such as database connection strings. To protect your credentials and prevent accidental leaks, the `.env` file is excluded from version control.

### 📦 Model & Encoder Access

The trained models and encoders required for running this project are not included in the repository due to storage limitations and versioning best practices.

➡️ **Please request access to the models and encoders via the link below:**

[📁 Request Access to Models on Google Drive](https://drive.google.com/file/d/1TuXW5fcYYc6kBvefKIfwu9TUjro9Ts7b/view?usp=sharing).

Once granted access, download and place the contents in the appropriate `data/` subfolders as described in the [Project Structure](#-project-structure).


 ### **Note:**  
- Remember to create your own `.env` file locally with the required configuration variables before running the application. Refer to the **Installation Instructions** section for the necessary environment variable setup.


## ▶️ Running Instructions

1. Start the backend API:

```bash
uvicorn backend.main:app --reload
```

2. Start the frontend Streamlit app:

```bash
streamlit run frontend/streamlit_app.py

```

3. Access the application:

- Open your browser and navigate to http://localhost:8501 for the Streamlit frontend.

- The backend API runs at http://localhost:8000 by default.


## 🙌 Contributors

- **Brahma Reddy Maddireddy** – *Neural Collabartive Filtering*|*Content-Based Filtering Model*|*Hybrid Model & Multi-Modal Fusion*

---


*Thank you for visiting and exploring the Freelancer Project Recommendation System!*


