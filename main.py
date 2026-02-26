import faker
import requests
import os
from dotenv import load_dotenv
from openrouter import OpenRouter
from faker import Faker


load_dotenv()
fake = faker.Faker()

SEARCH_API_KEY=os.getenv("SEARCH_API_KEY")
AI_API_KEY=os.getenv("AI_API_KEY")


client = OpenRouter(
    api_key=AI_API_KEY,
    server_url="https://ai.hackclub.com/proxy/v1",
)

def search_web(query: str) -> dict:
    response = requests.get(
        'https://search.hackclub.com/res/v1/web/search',
        params={'q': query, 'count': 5},
        headers={'Authorization': f'Bearer {SEARCH_API_KEY}'}
    )
    #print(f"Search API Status: {response.status_code}")
    #print(f"Search API Response: {response.text}")
    try:
        return response.json()
    except Exception as e:
        print(f"Failed to parse search response as JSON: {e}")
        raise


def ask_with_search(question: str) -> str:
    # Search the web
    results = search_web(question)

    # Format context
    context = "\n\n".join([
        f"[{r['title']}]({r['url']})\n{r['description']}"
        for r in results.get('web', {}).get('results', [])
    ])

    # Call via OpenRouter client (instead of requests.post)
    completion = client.chat.send(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Use these web results to answer accurately. "
                    "Respond In only 3 sentences. If information is conflicting then pick one of the results that do not conflict to summerize.\n\n"
                    f"{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    try:
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Failed to read AI completion: {e}")
        print(f"Raw completion object: {completion!r}")
        raise

if __name__ == "__main__":
    first_name=fake.first_name()
    last_name=fake.last_name()
    name=f"{first_name} {last_name}"
    answer=ask_with_search(f"Who is {name}")
    print(answer)