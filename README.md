# Volunteer and Community Service Recommender (CS467 Mini-Project)

> **DUY TAN UNIVERSITY**  
> **Course:** CS467 - Advanced Artificial Intelligence (2026-2027)  
> **Assignment 15:** Individual Mini-Project: NLP & Basics of Recommendation Systems  

---

## 📌 Project Identification
| Field | Details |
| :--- | :--- |
| **Student Name** | **Lê Văn Hoài Rin** |
| **Student ID** | **29211332113** |
| **Class / Section** | **CS467C (No. 14)** |
| **Instructor** | **Puri Vikram** |
| **Assigned Project Code** | **MP-C14** |
| **Assigned Project Title** | **Volunteer and Community Service Recommender** |
| **Application Domain** | **NGO and Student Civic Engagement** |
| **Repository URL** | [https://github.com/Rynne25/cs467-miniproject-29211332113](https://github.com/Rynne25/cs467-miniproject-29211332113) |

---

## 📖 1. Problem Statement
University students are increasingly eager to participate in civic engagement, non-governmental organization (NGO) initiatives, and community service projects. However, students frequently encounter difficulties finding opportunities that align with their specific schedules, personal skills, and social causes they genuinely care about (such as environmental protection, youth tutoring, healthcare, or disaster relief).

Existing volunteer portals typically rely on rigid keyword searches or chronological feeds, causing engagement fatigue and poor matching accuracy. This project builds an intelligent **Volunteer and Community Service Recommender System** that integrates:
1. **Natural Language Processing (NLP)** for text semantic understanding over volunteer activity descriptions, requirements, and past volunteer reflection notes.
2. **Hybrid Recommendation Engine** combining baseline popularity, semantic content-based filtering, and peer collaborative filtering with automated cold-start handling.

---

## 📊 2. Dataset Description (`data/`)

The project uses a domain dataset tailored to student civic engagement and volunteer management, stored in CSV format:

### 2.1 Activities Dataset (`data/items.csv`)
Contains **38 diverse volunteer activities** spanning 8 key social causes:
- `item_id`: Unique identifier (`ITEM_001` to `ITEM_038`).
- `activity_title`: Official title of the volunteer initiative.
- `cause_area`: Social impact domain (e.g., *Environmental Protection*, *Education & Literacy*, *Healthcare & Elderly Support*, *Hunger & Food Relief*, *Animal Welfare*, *Disaster Relief & Safety*, *Community Development*, *Youth & Sports Empowerment*).
- `schedule_type`: Time commitment (e.g., *Weekend Morning*, *Weekday Evening*, *Weekend All-Day*).
- `description`: Detailed paragraph outlining objectives, daily tasks, and community beneficiaries.
- `skill_tags`: Comma-separated domain tags and prerequisite skills (e.g., *beach cleanup, environmental education, kids mentoring, plastic waste sorting*).

### 2.2 Volunteer Ratings & Feedback Dataset (`data/ratings.csv`)
Contains **265 authentic volunteer interactions** from **15 active student volunteers** (`USR_01` to `USR_15`):
- `user_id`: Unique student volunteer identifier.
- `item_id`: Volunteer activity completed.
- `rating`: Rating on a 1.0 to 5.0 scale reflecting satisfaction and personal alignment.
- `reflection_note`: Short qualitative feedback describing their experience and takeaways.

### 2.3 Interaction Matrix Quality & Sparsity
- **Total Possible Pairs:** $15 \times 38 = 570$
- **Total Recorded Ratings:** $265$
- **Matrix Sparsity:** **$53.5\%$ missing entries** (perfectly within the realistic threshold of $40\%-65\%$).
- **Test User Persona (`USR_02`):** Specially curated as a returning education enthusiast who rated tutoring programs (`ITEM_004`, `ITEM_008`, `ITEM_015`, `ITEM_033`) with **5.0/5.0**.
- **Cold-Start Item (`ITEM_038`):** *"Riverside Community Book and Educational Toy Donation Drive"* has **0 ratings** in the database to rigorously test cold-start discovery.

---

## 🧠 3. NLP and Recommendation Methods

```
+---------------------------------------------------------------------------------------------+
|                                    RECOMMENDER ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------+
|                                                                                             |
|   [Free-Text Query]          [data/items.csv]               [data/ratings.csv]              |
|          │                          │                               │                       |
|          ▼                          ▼                               ▼                       |
|   ┌─────────────┐       ┌───────────────────────┐       ┌───────────────────────┐           |
|   │ Clean &     │       │ Enriched Text Corpus  │       │ User-Item Pivot       │           |
|   │ Tokenize    │       │ (Title + Cause + Desc)│       │ Interaction Matrix    │           |
|   └──────┬──────┘       └───────────┬───────────┘       └───────────┬───────────┘           |
|          │                          │                               │                       |
|          ▼                          ▼                               ▼                       |
|   ┌─────────────┐       ┌───────────────────────┐       ┌───────────────────────┐           |
|   │ Vectorize   │       │ TF-IDF Vectorizer     │       │ Pearson User          │           |
|   │ Query (1xD) │       │ (Unigram + Bigram)    │       │ Similarity Matrix     │           |
|   └──────┬──────┘       └───────────┬───────────┘       └───────────┬───────────┘           |
|          │                          │                               │                       |
|          ▼                          ▼                               ▼                       |
|   ┌─────────────────────────────────────────────┐       ┌───────────────────────┐           |
|   │ Cosine Similarity Match & Keyword Overlap   │       │ k-NN Rating           │           |
|   │ (Content-Based Engine)                      │       │ Prediction (User-CF)  │           |
|   └──────────────────────┬──────────────────────┘       └───────────┬───────────┘           |
|                          │                                          │                       |
|                          ▼                                          ▼                       |
|               ┌─────────────────────────────────────────────────────────────┐               |
|               │         WEIGHTED HYBRID ENGINE (α=0.55)                     │               |
|               │ Score = α * Sim_Content + (1-α) * Pred_CF_Norm              │               |
|               │ (Graceful Fallback to Content for Cold-Start Items & Users) │               |
|               └──────────────────────────────┬──────────────────────────────┘               |
|                                              │                                              |
|                                              ▼                                              |
|                                  [Top-5 Recommendations]                                    |
+---------------------------------------------------------------------------------------------+
```

### 3.1 NLP Feature Extraction (`src/nlp_pipeline.py`)
- **Text Preprocessing:** Lowercasing, punctuation stripping, regex normalization, stop-word elimination, and minimum token length filtering.
- **Enriched Corpus Representation:** Combines activity title (weighted), cause category (weighted), skill tags (weighted), and paragraph description.
- **Vector Space Model:** Fitted with `TfidfVectorizer` (sublinear term frequency scaling, unigram + bigram range `(1, 2)`, max 1,200 features).
- **User Interest Profile Modeling:** Aggregates TF-IDF vectors of activities rated $\ge 3.5$ stars by the student, weighted by satisfaction scores.
- **Explainability:** Identifies highest mutual TF-IDF term overlaps to provide matching keyword rationale for each recommendation.

### 3.2 Recommendation Approaches (`src/recommender.py`)
1. **Popularity Baseline:** Bayesian average weighted rating score:
   $$W = \frac{R \cdot v + C \cdot m}{v + m}$$
   where $R$ is average rating, $v$ is rating count, $m$ is minimum threshold ($m \ge 2$), and $C$ is global mean. Filters out items already completed by the target volunteer.
2. **NLP Content-Based Recommender:**
   - **User Profile Mode:** Computes $\text{CosineSimilarity}(\vec{u}_{\text{profile}}, \vec{v}_{\text{item}})$, recommending Top-5 unseen activities.
   - **Natural Language Query Search:** Projects free-text queries $\vec{q}$ into the shared vocabulary space to rank Top-5 activities with similarity scores and keyword explanations.
3. **User-Based Collaborative Filtering (User-CF):** Centered cosine (Pearson correlation) across volunteers with $k$-nearest neighbor aggregation ($k=5$).
4. **Weighted Hybrid Recommender:** Linearly blends content similarity with normalized collaborative filtering scores:
   $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{content}} + (1 - \alpha) \cdot \text{Score}_{\text{CF}}$$
   *Cold-Start Resiliency:* If an item has 0 ratings (or user has no rating history), the engine automatically falls back to pure content-based matching without crashing.

---

## 💻 4. Installation & Run Commands

### 4.1 Prerequisites & Setup
Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/Rynne25/cs467-miniproject-29211332113.git
cd cs467-miniproject-29211332113

# Install required Python packages
pip install -r requirements.txt
```

### 4.2 Running the Application
```bash
# 1. Run the automated demo (Executes all 3 test cases & evaluation benchmark)
python src/main.py

# 2. Run offline evaluation benchmark and generate charts into results/
python src/main.py --eval

# 3. Test a custom natural language query
python src/main.py --query "weekend beach cleanup and environmental education for kids"

# 4. Get personalized hybrid recommendations for a returning volunteer
python src/main.py --user USR_02

# 5. Launch interactive CLI prompt
python src/main.py --interactive
```

---

## 🧪 5. Sample Outputs for Required Test Cases

Detailed log is saved in `results/sample_outputs.txt`.

### Test Case 1: Natural Language Interest Query Search
**Query:** *"weekend beach cleanup and environmental education for kids"*
```
  Rank 1: [ITEM_001] Weekend Coastal Beach Cleanup and Marine Conservation
          Cause Area       : Environmental Protection
          Cosine Score     : 0.5298
          Matching Keywords: beach cleanup, beach, cleanup, cleanup environmental

  Rank 2: [ITEM_012] Junior Eco-Warrior Beach Cleanup and Nature Discovery
          Cause Area       : Environmental Protection
          Cosine Score     : 0.4123
          Matching Keywords: beach cleanup, beach, cleanup, kids

  Rank 3: [ITEM_021] Coastal Mangrove Restoration and Children Nature Tour
          Cause Area       : Environmental Protection
          Cosine Score     : 0.1917
          Matching Keywords: kids, education kids, environmental education, environmental

  Rank 4: [ITEM_004] Weekend STEM and Math Tutoring for Underprivileged Kids
          Cause Area       : Education & Literacy
          Cosine Score     : 0.1473
          Matching Keywords: kids, education kids, weekend, education

  Rank 5: [ITEM_015] Elementary Science Discovery and Homework Help Tutoring Lab
          Cause Area       : Education & Literacy
          Cosine Score     : 0.1040
          Matching Keywords: education kids, kids, education
```

### Test Case 2: Returning Volunteer Recommendation (`USR_02`)
**Profile:** Volunteer `USR_02` has previously awarded 5.0/5.0 stars to academic tutoring and literacy programs (`ITEM_004`, `ITEM_008`, `ITEM_015`, `ITEM_033`).
```
  Rank 1: [ITEM_028] Coding and Robotics Mentorship for High School Students
          Cause Area       : Education & Literacy
          Hybrid Score     : 0.6647 (Hybrid (Content + CF))
          Profile Overlap  : tutoring, literacy, education literacy

  Rank 2: [ITEM_027] Community Solar Oven Workshop and Renewable Energy Fair
          Cause Area       : Environmental Protection
          Hybrid Score     : 0.6495 (Hybrid (Content + CF))
          Profile Overlap  : science, education, stem

  Rank 3: [ITEM_017] Feline Foster Care and Cat Adoption Weekend Fair
          Cause Area       : Animal Welfare
          Hybrid Score     : 0.4768 (Hybrid (Content + CF))
          Profile Overlap  : public, weekend, foster

  Rank 4: [ITEM_035] Public Park Native Birdhouse and Bat Sanctuary Building
          Cause Area       : Environmental Protection
          Hybrid Score     : 0.4499 (Hybrid (Content + CF))
          Profile Overlap  : education, public, environmental

  Rank 5: [ITEM_021] Coastal Mangrove Restoration and Children Nature Tour
          Cause Area       : Environmental Protection
          Hybrid Score     : 0.4200 (Hybrid (Content + CF))
          Profile Overlap  : education, kids, education kids
```

### Test Case 3: Cold-Start Item Retrieval (`ITEM_038` with 0 Ratings)
**Item:** `ITEM_038` - *"Riverside Community Book and Educational Toy Donation Drive"* (Total prior ratings = 0).  
**Test Query:** *"rural community donation drive collecting children books and toys"*
```
  Rank 1: [ITEM_038] Riverside Community Book and Educational Toy Donation Drive [TARGET COLD-START ITEM!]
          Cause Area       : Community Development
          Cosine Score     : 0.5462
          Matching Keywords: donation drive, donation, drive, toys

  Rank 2: [ITEM_006] Community Food Bank Packaging and Distribution Drive
          Cosine Score     : 0.0745
          Matching Keywords: drive, community

  Rank 3: [ITEM_025] Rural Mobile Dental and Health Screening Support
          Cosine Score     : 0.0663
          Matching Keywords: rural

  Rank 4: [ITEM_027] Community Solar Oven Workshop and Renewable Energy Fair
          Cosine Score     : 0.0288
          Matching Keywords: community, children

  Rank 5: [ITEM_030] Sensory Garden Maintenance for Children with Special Needs
          Cosine Score     : 0.0279
          Matching Keywords: children
```
*Result:* **SUCCESS** — Cold-start item is matched and ranked #1 despite having 0 prior interaction records.

---

## 📈 6. Evaluation Results Table & Visualizations

### 6.1 Quantitative Offline Benchmark (Holdout 20% per volunteer)
| Recommendation Approach | Precision@5 | Recall@5 | HitRate@5 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Popularity Baseline** | 0.0800 | 0.1667 | 0.4000 | Good baseline but ignores personalized domain interests |
| **User-Based Collaborative Filtering** | 0.0533 | 0.1333 | 0.2667 | Leverages volunteer consensus; struggles on sparse items |
| **NLP Content-Based Recommender** | 0.1867 | 0.4444 | 0.7333 | Accurately matches interest vectors; immune to item cold-start |
| **Weighted Hybrid Recommender** | **0.1600** | **0.3889** | **0.6000** | **Balances semantic fit and peer consensus with smooth cold-start fallback** |

### 6.2 Visualizations (`results/`)
1. **`results/data_distribution.png`**: Illustrates rating frequencies (1 to 5 stars) and distribution of activities across the 8 civic cause domains.
2. **`results/model_comparison.png`**: Bar chart comparing Precision@5, Recall@5, and HitRate@5 across all four recommendation methods.

---

## 🔍 7. Short Reflection

### What Worked Well:
- **Semantic Text Matching:** The TF-IDF sublinear vectorizer with unigram + bigram representations captured nuanced phrases (such as *"beach cleanup"*, *"math tutoring"*, *"environmental education"*) with high discriminative power.
- **Hybrid Synergy:** Combining Content-Based profiling with User-CF improved Precision@5 from 0.0800 (Popularity) to **0.1600** (Hybrid), while achieving a **60.0% HitRate@5** and providing a balanced recommendation experience.
- **Cold-Start Resilience:** New volunteer initiatives (`ITEM_038`) with zero historical ratings can be recommended via content and query matching.

### Limitations & Future Work:
- **TF-IDF Vocabulary Sparsity:** TF-IDF cannot recognize semantic synonyms (e.g., *"ocean"* vs. *"marine"*, *"pupil"* vs. *"student"*). Future iterations could adopt dense sentence transformer embeddings (e.g., `all-MiniLM-L6-v2`).
- **Dynamic User Feedback:** The current profile aggregates past ratings statically. Incorporating time decay or active session interaction would further improve real-time recommendation freshness.

---

## 📂 Project Directory Structure

```
cs467-miniproject-29211332113/
├── README.md                 # Full project documentation & report
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data/
│   ├── items.csv             # 38 volunteer activities with text descriptions and skill tags
│   └── ratings.csv           # 265 volunteer ratings and reflection notes (53.5% sparsity)
├── src/
│   ├── nlp_pipeline.py       # Text preprocessing, TF-IDF vectorizer, user interest profile
│   ├── recommender.py        # Popularity, Content-Based, User-CF, Weighted Hybrid
│   ├── evaluate.py           # Offline evaluation metrics and chart plotting
│   └── main.py               # Runnable CLI entry point and test case driver
├── notebooks/
│   └── demo.ipynb            # Interactive walkthrough Jupyter notebook
└── results/
    ├── sample_outputs.txt    # Saved console outputs of all 3 mandatory test cases
    ├── data_distribution.png # Distribution of ratings and cause areas
    └── model_comparison.png  # Performance comparison bar chart
```

---
*Submitted for CS467 Advanced Artificial Intelligence - Duy Tan University, Academic Year 2026-2027.*
