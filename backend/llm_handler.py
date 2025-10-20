from config import settings
from groq import Groq

class LLMHandler:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq"""
        
        system_prompt = """You are a helpful assistant that answers questions ONLY based on the provided context.
Do NOT use any external knowledge or information not in the context.
Do NOT make up citations.
If the answer is not in the context, say: "I cannot find this information in the provided documents."
Always cite sources using [1], [2], etc. that correspond to the numbered context items."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer using ONLY the provided context. Do not add external knowledge."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            raise

llm_handler = LLMHandler()