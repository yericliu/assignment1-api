# Assignment 1

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

GET `/embedding?word=automobile` returns the word vector.
