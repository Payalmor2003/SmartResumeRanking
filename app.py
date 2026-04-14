from flask import Flask, render_template, request
import os
from utils import (
    extract_text_from_pdf, clean_text, rank_resumes,
    find_missing_skills, find_matched_skills,
    compute_skill_score, compute_final_score,
    get_hire_probability, extract_experience_years
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/rank', methods=['POST'])
def rank():
    job_description_raw = request.form['job_description']
    job_description_clean = clean_text(job_description_raw)

    uploaded_files = request.files.getlist("resumes")

    resumes_raw = []
    resumes_clean = []
    filenames = []

    for file in uploaded_files:
        if file.filename == '':
            continue
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        raw_text = extract_text_from_pdf(filepath)
        cleaned = clean_text(raw_text)

        resumes_raw.append(raw_text)
        resumes_clean.append(cleaned)
        filenames.append(file.filename)

    if not resumes_clean:
        return render_template("index.html", error="No valid resumes uploaded.", show_form=True)

    tfidf_scores = rank_resumes(job_description_clean, resumes_clean)

    results = []
    for i in range(len(filenames)):
        matched = find_matched_skills(job_description_raw, resumes_raw[i])
        missing = find_missing_skills(job_description_raw, resumes_raw[i])

        skill_score = compute_skill_score(matched, missing)
        tfidf_pct = round(tfidf_scores[i] * 100, 2)
        final_score = compute_final_score(tfidf_pct, skill_score)

        hire_label, hire_color = get_hire_probability(final_score)
        experience = extract_experience_years(resumes_raw[i])

        # Clean filename for display
        display_name = filenames[i].replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()

        results.append({
            "filename": filenames[i],
            "display_name": display_name,
            "score": final_score,
            "tfidf_score": tfidf_pct,
            "skill_score": skill_score,
            "matched": matched,
            "missing": missing,
            "hire_label": hire_label,
            "hire_color": hire_color,
            "experience": experience,
            "match_count": len(matched),
            "missing_count": len(missing),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Add rank position
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return render_template("index.html", results=results, show_form=True)

if __name__ == "__main__":
    app.run(debug=True)
