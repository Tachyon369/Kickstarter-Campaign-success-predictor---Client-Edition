# Kickstarter Campaign Success Predictor

**Group Number:** 10  
**Hugging Face Demo:** [Kickstarter Campaign Success Predictor](https://huggingface.co/spaces/Tachyon99/Kickstarter_Campaign_Success_Predictor)

---

## 👥 Team Members

- **MBA/0168/61** — Sangadi Rakendu Rajvallabh  
- **MBA/0174/61** — Arjun P P  
- **MBA/0198/61** — Konduru Ranadev Varma  
- **MBA/0060/61** — Roopal Singh  
- **MBA/0163/61** — Abhijeet Jaswant  

---

## 📂 Project Structure

### **`UI/`**
- Contains the Hugging Face application files.
- Includes LightGBM model files used for prediction.

### **`Trainer/`**
- UI for training custom models.
- Allows the user to:
  - Select the type of model.
  - Choose relevant data for training.

### **IPYNB Files**
- Data cleaning and preprocessing.
- Implementation of the LightGBM model used in the final deployment.
- Initially used **Llama.cpp** for faster inference, but due to recent Hugging Face installation issues (past 3 days), switched to **GPT4All** (slightly lower performance).

---

## 🛠️ How to Run

### **If Cloning the Repository**
#### **Windows**
- Use the provided `.bat` files to run:
  - **Trainer**
  - **UI**

#### **Linux / macOS**
- Run the respective `app.py` files directly.

---

## 📌 Requirements
- **Python Version:** `3.12.9`
- Install dependencies via:
```bash
pip install -r requirements.txt
