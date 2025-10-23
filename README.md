## 🎬 Demo Video
[Watch on Google Drive]([https://drive.google.com/your-link](https://drive.google.com/file/d/1v9AZtGodlR_osXfVovAeGfPQIa5q5G-k/view?usp=drive_link))
## Trained Model 
[Google Drive]([https://drive.google.com/file/d/1ZIiFS6xiTEtudsP3hQk7iTf77dWHnnmH/view?usp=sharing])



---

## Features

1. **User Interface (Gradio)**  
   - Accepts customer queries and predicts intent.
   - Fast and interactive frontend.
   - Internally calls **FastAPI** endpoints to query the model.

2. **Model (DistilBERT)**  
   - Pretrained on a large 31k airline dataset.
   - Supports incremental retraining using feedback.

3. **Admin Panel**  
   - Review user feedback and mark correct/incorrect predictions.
   - Automatic retraining on corrected feedback.
   - Keeps a subset of original data to avoid forgetting previous knowledge.

4. **Feedback-based Learning**  
   - Collects feedback in `feedback.csv`.
   - Retrains model only when enough feedback is accumulated.
   - Automatically resets feedback after retraining.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ASAPP
