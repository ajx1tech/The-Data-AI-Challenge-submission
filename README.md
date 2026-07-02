# Redrob AI - Intelligent Candidate Discovery & Ranking Challenge

## Architecture Overview
Our solution is a robust, purely CPU-based Semantic + Heuristic pipeline that strictly conforms to the computational constraints (Max 5 mins, Max 16GB RAM, No Network).

```text
[ candidates.jsonl.gz ] (100k)        [ job_description.md ]
          |                                     |
    +-----+-----+                         +-----+-----+
    | Extractor |                         | JD Parser |
    +-----+-----+                         +-----+-----+
          |                                     |
    [ Raw Texts ]                         [ Context ]
          |                                     |
+---------+---------+                 +---------+---------+
| Sentence Embedder |                 | Sentence Embedder |
| (all-MiniLM-L6)   |                 | (all-MiniLM-L6)   |
+---------+---------+                 +---------+---------+
          |                                     |
    [ Vector DB (FAISS) ] <=====================>
          |
    [ Top 500 Cands ]
          |
+---------+---------+
| Honeypot Detector | -> Discards invalid profiles
+---------+---------+
          |
+---------+---------+
| Candidate Scorer  | -> Applies 4-category weighted scoring
+---------+---------+
          |
    [ Sorted 100 ]
          |
+---------+---------+
| Output Generator  | -> Enforces formatting constraints & generates reasoning
+---------+---------+
          |
  [ submission.csv ]
```

## Quick Start
1. Place the dataset in the data folder: `data/candidates.jsonl.gz`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the ranking pipeline:
   ```bash
   python rank.py
   ```
4. Verify compliance:
   ```bash
   python validate_submission.py
   ```
5. Convert to Excel format:
   ```bash
   python create_xlsx.py
   ```

## Compute Compliance Checklist
- [x] Execution Time < 5 minutes (FAISS + MiniLM is highly optimized for CPU inference)
- [x] Memory < 16 GB RAM (Batch processing and generator patterns utilized where possible)
- [x] Hardware: Pure CPU (No CUDA calls)
- [x] Network: OFF (No OpenAI API calls; reasonings are template-based)
- [x] Output: Exactly 100 rows, proper tie-breaking, bounded scores.

## Scoring Formula
Final Score = `(Skills (35%) + Exp (25%) + Behavioral (25%) + Logistics (15%)) * Disqualifier Multipliers - Red Flag Penalties`