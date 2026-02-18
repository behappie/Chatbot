
import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Any

load_dotenv()

async def test_streaming_speed():
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # Using Gemini 2.0 Flash for maximum speed
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = "Explain the concept of Opportunity Cost in 3 paragraphs for a Singapore JC student. Use British English."
    
    print(f"Testing streaming speed with model: {model.model_name}")
    print(f"Prompt: {prompt}\n")
    
    start_time = datetime.now().timestamp()
    first_chunk_time = None
    full_text = ""
    
    print("--- Streaming Start ---")
    response = await model.generate_content_async(prompt, stream=True)
    async for chunk in response:
        if first_chunk_time is None:
            first_chunk_time = datetime.now().timestamp()
            print(f"Time to first chunk: {first_chunk_time - start_time:.4f}s")
        
        print(chunk.text, end="", flush=True)
        full_text += chunk.text
    
    end_time = datetime.now().timestamp()
    print("\n--- Streaming End ---")
    
    print(f"\nTotal generation time: {end_time - start_time:.4f}s")
    print(f"Total characters: {len(full_text)}")
    print(f"Average speed: {len(full_text) / (end_time - start_time):.2f} chars/sec")

if __name__ == "__main__":
    asyncio.run(test_streaming_speed())
