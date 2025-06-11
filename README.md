# TDS Virtual TA

A virtual Teaching Assistant API for the Tools in Data Science course at IIT Madras Online Degree Program.

## Features

- Answers student questions based on course content and Discourse posts
- Supports multimodal queries (text + images)
- Uses RAG (Retrieval Augmented Generation) for accurate answers
- Provides source links for answers

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
API_KEY=your_aipipe_api_key_here
```

4. Prepare the knowledge base:
```bash
python preprocess.py
```

5. Run the server:
```bash
uvicorn app:app --reload
```

## API Usage

### Query Endpoint

`POST /query`

Request body:
```json
{
  "question": "Your question here",
  "image": "optional_base64_image"
}
```

Response:
```json
{
  "answer": "The answer to your question",
  "links": [
    {
      "url": "source_url",
      "text": "reference text"
    }
  ]
}
```

### Health Check

`GET /health`

Returns the health status of the service and database statistics.

## License

MIT License - See LICENSE file for details
