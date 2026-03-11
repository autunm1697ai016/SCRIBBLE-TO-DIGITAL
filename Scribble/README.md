# 📝 Scribble to Digital

Messy Notes Image → Clean Text + To-Do List

## 🔧 Tech Stack

- **Python** – Core language
- **Streamlit** – UI framework
- **OpenCV** – Image enhancement
- **Pytesseract** – OCR text extraction
- **GenAI API** (OpenAI) – Context correction & task extraction

## 📂 Project Structure

```
scribble_to_digital/
│
├── app.py
├── requirements.txt
├── utils.py
└── README.md
```

## 📦 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** You also need Tesseract OCR installed on your system:
- **Windows:** Download from [GitHub - UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt-get install tesseract-ocr`

## 🔑 Set GenAI API Key

### Windows (PowerShell)
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

### Mac/Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🧠 Functionalities Breakdown

### 1️⃣ OCR Enhancement (Brighten + Clean Image)
- Improves messy notes before OCR processing
- Uses OpenCV to enhance contrast and clarity
- Applies binary thresholding for optimal text extraction

### 2️⃣ Contextual Logic (GenAI)
- Fixes wrong words detected by OCR
- Repairs broken sentences
- Corrects spelling errors using context

### 3️⃣ To-Do Extraction
- Separates tasks from general notes
- Identifies action items
- Organizes output in a structured format

## ▶️ Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📝 How It Works

1. **Upload** a messy handwritten notes image
2. **View** the original and enhanced versions
3. **Extract** raw OCR text
4. **Click** "Convert to Digital" to:
   - Clean the text using AI
   - Extract and organize to-do items
   - Generate formatted output

## 🛠️ File Descriptions

- **app.py** – Main Streamlit application with UI and workflow
- **utils.py** – Helper functions for image enhancement and text extraction
- **requirements.txt** – Python dependencies

---

Made with ❤️ for cleaner notes!
