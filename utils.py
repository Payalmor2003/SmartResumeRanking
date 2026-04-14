import PyPDF2
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Multi-word and single-word tech skills
TECH_SKILLS = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering",
    "artificial intelligence", "neural networks", "reinforcement learning",
    "large language models", "generative ai",

    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "swift", "kotlin", "scala", "r",

    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "opencv",

    "flask", "django", "fastapi", "spring", "express", "node.js",
    "react", "angular", "vue", "next.js",

    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "oracle",

    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "jenkins", "ci/cd", "github actions",

    "html", "css", "tailwind", "bootstrap",
    "linux", "git", "github", "rest api", "graphql", "microservices",

    "nlp", "llm", "bert", "gpt", "transformers", "hugging face",
    "langchain", "vector database", "rag",

    "excel", "tableau", "power bi", "looker",
    "agile", "scrum", "jira", "devops",
]

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def clean_text(text):
    """Normalize and clean text."""
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)  # keep +, #, . for C++, C#, .NET
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words or len(w) <= 2]
    return " ".join(words)

def find_matched_skills(job_desc_raw, resume_text_raw):
    """Find skills that appear in both job description and resume."""
    job = job_desc_raw.lower()
    resume = resume_text_raw.lower()
    matched = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, job) and re.search(pattern, resume):
            matched.append(skill)
    return matched

def find_missing_skills(job_desc_raw, resume_text_raw):
    """Find skills required by job but absent in resume."""
    job = job_desc_raw.lower()
    resume = resume_text_raw.lower()
    missing = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, job) and not re.search(pattern, resume):
            missing.append(skill)
    return missing

def rank_resumes(job_description_clean, resumes_clean):
    """TF-IDF cosine similarity ranking."""
    documents = [job_description_clean] + resumes_clean
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return scores

def compute_skill_score(matched, missing):
    """Score based on skills match ratio."""
    total = len(matched) + len(missing)
    if total == 0:
        return 0.0
    return round((len(matched) / total) * 100, 2)

def compute_final_score(tfidf_score, skill_score):
    """Weighted blend: 60% TF-IDF + 40% skill score."""
    return round((tfidf_score * 0.6) + (skill_score * 0.4), 2)

def get_hire_probability(score):
    """Classify hire probability based on final score."""
    if score >= 75:
        return "High", "#22c55e"
    elif score >= 50:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#ef4444"

def extract_experience_years(text):
    """Try to find years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s*experience',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None
