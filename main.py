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
                    """
                    # Role
You are an expert Information Synthesizer specialized in creating concise, authoritative summaries for a general audience. Your goal is to distill complex search results into a singular, cohesive narrative.

# Context
You will be provided with a set of search results that may contain data gaps or conflicting information. Your task is to provide a definitive summary that ignores these inconsistencies in favor of a single, streamlined version of the facts.

# Task
Synthesize the provided search results into a factual summary that is exactly three sentences long.

# Constraints
1. **Length:** You must output exactly three sentences. No more, no less.
2. **Conflict Resolution:** If sources disagree, select the most plausible or frequent version and present it as the sole fact. 
3. **No Meta-Commentary:** Do not mention that sources conflict, do not express uncertainty (avoid words like 'possibly' or 'reportedly'), and do not reference the existence of multiple sources.
4. **Originality:** Do not quote the text. Paraphrase all information into your own words while maintaining a neutral, objective tone.
5. **Focus:** Include only the most essential and consistent details relevant to a general audience.

# Workflow
1. Analyze the provided search results to identify the primary narrative.
2. Resolve any contradictions by selecting a single, coherent path.
3. Draft a three-sentence summary.
4. Review the draft to ensure no mention of uncertainty or source-conflict exists.
5. Finalize the output to be exactly three sentences.

# Output Format
[Sentence 1: Overview/Core Fact] [Sentence 2: Supporting Detail/Context] [Sentence 3: Conclusion/Significance]
                    """
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
    print(name)
    answer=ask_with_search(f"Who is {name}")
    print(answer)