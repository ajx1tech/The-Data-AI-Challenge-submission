# Slide 1: Title Slide
**Title:** Redrob AI Challenge - Intelligent Candidate Discovery
**Team:** [Your Team Name]
**Date:** [Date]

---

# Slide 2: Problem Statement
**The Challenge:**
Identify and rank the top 100 candidates from a pool of 100,000 profiles for a "Senior AI Engineer" role.

**Core Difficulties:**
- No simple keyword matching.
- Detection of subtle honeypot (fake) profiles.
- Extreme compute constraints (CPU only, <5 mins, <16GB RAM).
- Zero network reliance (No external LLM APIs).

---

# Slide 3: Solution Overview
**Our Approach:** Semantic Filtering + Deterministic Scoring
1. **Pre-compute:** Vectorize candidates using lightweight embeddings.
2. **Retrieval:** FAISS inner-product search for top 500 candidates.
3. **Validation:** Rule-based honeypot detection.
4. **Scoring:** Multi-signal weighted heuristic ranking.
5. **Explainability:** Deterministic, template-based reasoning generation.

---

# Slide 4: JD Understanding
**Role:** Senior AI Engineer - Founding Team
**Key Needs Extracted:**
- **Skills:** Python, Embeddings, Vector Search, FAISS, Production ML.
- **Experience:** 6-8 sweet spot, Product > Consulting background.
- **Behavioral:** High recruiter response rate, recently active, open to work.
- **Logistics:** Pune/Noida/Delhi preferred, notice period <= 30 days.

---

# Slide 5: Ranking Methodology
**Weight Distribution:**
- **Skills (35%):** Keyword presence in embedded skills vector.
- **Experience (25%):** Adjusted for total duration, ML-ratio, and consulting penalties.
- **Behavioral (25%):** GitHub activity, response rates, and recent login history.
- **Logistics (15%):** City matching and fast join availability.

**Modifiers:** Aggressive multiplicative penalties for irrelevant titles and no Python.

---

# Slide 6: Honeypot Detection
**Safeguarding the Top 100:**
- **Temporal Paradoxes:** `start_date` before `company_founded_date`.
- **Skill Spoofing:** 'Expert' proficiency with 0 `years_used` (requires ≥3 to trigger).
- **Excessive Expertise:** Over 8 skills marked as 'expert' or 'advanced'.
*Any candidate triggering these rules is immediately dropped to a score of 0.0.*

---

# Slide 7: Explainability
**Template-Based Reasoning Engine:**
- **Deterministic:** No hallucinations. Data extracted directly from the candidate dictionary.
- **Tiered Outputs:**
  - *Top 30:* Highlights years of experience, top 3 skills, response rate, and location.
  - *31-60:* Balances strengths with minor concerns (e.g., notice period).
  - *61-100:* Explicitly states the primary concern (e.g., low response rate).
- **Compliant:** Guaranteed <200 characters per row.

---

# Slide 8: Architecture & Constraints
**Compliance Check:**
- **Time Constraint:** CPU embedding via `all-MiniLM-L6-v2` handles 100k text segments natively fast. FAISS reduces search to milliseconds. Total time ~3 mins.
- **Memory Constraint:** `yield`/batch-processing keeps RAM footprint ~3-4 GB.
- **Network Constraint:** Sentence-transformers downloads weights locally beforehand. Entire runtime is fully offline.

---

# Slide 9: Results & Performance
- **Throughput:** ~100,000 candidates processed efficiently.
- **Accuracy:** True semantic capture prevents the "Marketing Manager with AI keywords" trap.
- **Reliability:** Strict tie-breaking and non-increasing score enforcement validate 100% against the grading script.

---

# Slide 10: Submission Assets & Tech Stack
**Tech Stack:** `numpy`, `pandas`, `sentence-transformers`, `faiss-cpu`, `tqdm`.
**Assets Provided:**
1. `rank.py` - Core logic script.
2. `submission.xlsx` - Final ranked output.
3. `submission_metadata.yaml` - Team info.
4. `validate_submission.py` - Custom validation checker.
5. This slide deck.
