import spacy
import logging
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import process
import fitz  # PyMuPDF for PDF extraction
import docx
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st

# --- Pre-initialize NLP model ---
def init_nlp():
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])  # Optimized pipeline
        logger.info("NLP model loaded successfully with optimized pipeline.")
        return nlp
    except Exception as e:
        logger.error(f"Error loading NLP model: {str(e)}")
        raise

nlp = init_nlp()

# --- Load skill list ---
def load_skill_list():
    with open("skills_list.txt", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

skills_gazetteer = load_skill_list()

# --- Helper Functions ---
def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    content = uploaded_file.read()  # Read once and store
    uploaded_file.seek(0)  # Reset file pointer
    if ext == "pdf":
        try:
            pdf = fitz.open(stream=content, filetype="pdf")
            return "".join(page.get_text() for page in pdf)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""
    elif ext == "docx":
        try:
            docf = docx.Document(BytesIO(content))
            return "\n".join([para.text for para in docf.paragraphs])
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return ""
    elif ext == "txt":
        return content.decode("utf-8", errors='replace')
    st.warning(f"Unsupported file type: {ext}")
    return ""

@st.cache_data(show_spinner=False)
def extract_skills_enhanced(text):
    if not text or all(c in "•\n\t " for c in text.strip()):
        return []
    text = text[:5000] if len(text) > 5000 else text  # Early size reduction
    text_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]  # 1,000-char chunks
    all_skills = set()
    for chunk in text_chunks:
        doc = nlp(chunk)
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(skill) for skill in skills_gazetteer if len(skill.split()) <= 3]
        matcher.add("SKILL", patterns)
        matches = matcher(doc)
        matcher_skills = set([doc[start:end].text.lower() for _, start, end in matches])
        all_skills.update(matcher_skills)
    
    # Limited fuzzy matching for missed skills
    found_gazetteer = set()
    words = [w for w in text.lower().split() if w.strip()]
    for n in [1]:  # Limit to 1-grams
        for i in range(min(100, len(words) - n + 1)):  # Limit to first 100 words
            ng = " ".join(words[i:i + n])
            match, score = process.extractOne(ng, skills_gazetteer)
            if score and score > 95:
                found_gazetteer.add(match)
    
    all_skills.update(found_gazetteer)
    nice_skills = [s for s in all_skills if len(s) > 2]
    return sorted(set(nice_skills))[:15]

def extract_profile_skills(user_id, file=None, text=None, api_base_url="http://127.0.0.1:8000"):
    if file:
        user_input_text = extract_text_from_file(file)
    elif text:
        user_input_text = text
    else:
        return []
    # with st.spinner("Extracting skills..."):
    skills = extract_skills_enhanced(user_input_text)
    if skills:
        try:
            if file:
                files = {"file": (file.name, file, file.type)}
                response = requests.post(
                    f"{api_base_url}/profile/extract/{user_id}",
                    files=files,
                    timeout=30
                )
            else:
                response = requests.post(
                    f"{api_base_url}/profile/extract/{user_id}",
                    data={"skills": ", ".join(skills)}
                )
            if response.status_code == 200:
                return skills
            # else:
            #     st.error(f"Failed to extract skills: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"API error: {str(e)}")
    return skills