# Steps
Make a virtual python envrionment to avoid issues


run app
```shell
uvicorn main:app --reload
```

deployed at http://127.0.0.1:8000/
docs at http://127.0.0.1:8000/docs

# Steps
1. run `webscraper.py` to get the tournament instructions PDF and the registration form.
2. run `download.py` to download the instructions pdf file




# References
RAG stuff: https://colab.research.google.com/drive/1pNKEXX_f7MwovPyrh-uMbE-UtFIxZbik?authuser=1#scrollTo=izRCxPini7F_ (requires UCSC email to access)

Using OpenRouter: https://colab.research.google.com/drive/12UKc9ZmwfwgwnRVDuKBfOQlBQjJqhMC4#scrollTo=BNYyDCNuAhbj
- OpenRouter has several models that can be used for FREE!
- Given the usage of RAG, the result of the prompts is the same as we give it the same information.
- Given that the information is relatively simple, free models perform just as well as paid ones

Use OpenRouter information in the OpenAI client
```python
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
```

using this model: `meta-llama/llama-3.2-11b-vision-instruct:free`


Options
https://openrouter.ai/models?fmt=cards&input_modalities=text&max_price=0&order=throughput-high-to-low&q=llama%20instruct

We can use this model?
https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free


make `.env` file like this:
```
API_KEY=[google search API key]
SEARCH_ENGINE_ID=[google search engine id]
OPENAI_API_KEY=[openrouter API key]
OPENAI_API_BASE=https://openrouter.ai/api/v1 (use this regardless of OpenAI model)
OPENAI_API_MODEL_NAME=meta-llama/llama-3.2-11b-vision-instruct:free (or any other model. for our use case, a free model performs just as well as ChatGPT 4.1, which is what I was using before)
```

encountering strange issues with openrouter
```
Error processing question 'What date does registration open?': 500: Error processing question: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-4b-04-28/2e98edb5-b21b-455b-afb4-d5c01aad515d. High demand for qwen/qwen3-4b:free on OpenRouter - limited to 1 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '1', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1752248580000'}, 'provider_name': None}}, 'user_id': '...'}
```

need to purchase credits to increase rate limits
https://openrouter.ai/docs/api-reference/limits

activate virtual environment
```
source .venv/bin/activate
```

get a firebase json from python and add stuff to the env file.