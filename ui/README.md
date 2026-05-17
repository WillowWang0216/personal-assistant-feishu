# personal-assistant UI

Memory management visualization interface, built on Streamlit.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
cd ui
streamlit run app.py
```

The app will open in a browser at `http://localhost:8501`.

## Features

- **All Memories**: View all active memories in PersonalMemoryStore, filterable by type
- **Search Test**: Input query terms to test BM25+7-factor retrieval, adjust result count
- **Statistics**: View memory store size, event history
- **Configuration**: View current retrieval weights and system configuration

## Do I Need to Keep It Running?

No. The UI runs independently, only reading the shared SQLite database and config file. It does not affect the main Agent process. Start it when needed, close it when not.
