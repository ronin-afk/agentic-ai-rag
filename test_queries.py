import requests


URL = "http://127.0.0.1:8000/chat"


queries = [
    "What is Agentic AI?",

    "What are the main architectural components "
    "required to build agentic systems?",

    "What real-world industry use cases "
    "for Agentic AI are discussed in the eBook?",

    "How does Agentic AI differ from traditional "
    "generative AI chatbots according to the text?",

    "What key challenges or limitations of "
    "Agentic AI are mentioned in the document?",

    "What is the capital of France?",
]


for query in queries:

    response = requests.post(
        URL,
        json={"query": query}
    )

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(query)

    print("\nRESPONSE:")
    print(response.json())