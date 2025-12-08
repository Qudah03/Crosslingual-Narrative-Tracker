import ollama
import logging

# Configure logging
logger = logging.getLogger(__name__)

class LocalSummarizer:
    def __init__(self, model_name="phi3"):
        self.model = model_name

    def summarize_cluster(self, headlines):
        """
        Uses local LLM to summarize a list of headlines into one sentence.
        """
        # 1. Prepare the prompt
        # We limit to top 10 headlines to save context window
        bullet_points = "\n".join([f"- {h}" for h in headlines[:10]])
        
        prompt = f"""
        You are a neutral news aggregator. 
        Read the following headlines and write a SINGLE sentence summary of the event.
        Do not use flowery language. Be direct.
        
        HEADLINES:
        {bullet_points}
        
        SUMMARY:
        """

        try:
            # 2. Call Ollama (Running locally on your RTX 4050)
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            # 3. Extract text
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Summary unavailable (LLM Error)"

# Quick test block
if __name__ == "__main__":
    test_headlines = [
        "Gold prices hit record high amid market uncertainty",
        "Investors flock to safe-haven assets as gold soars",
        "Economic downturn fears push gold past $2500"
    ]
    summarizer = LocalSummarizer()
    print("Test Summary:", summarizer.summarize_cluster(test_headlines))