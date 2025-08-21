- #agentic_ai
- > Article wriiten by AI and Driven by Human
- # Building a Memory-Enabled Agent with Strands and Ollama
  
  In this post, we’ll walk through how to add **memory** to your first Strands agent.  
  By the end, your agent will be able to **remember, recall, and forget** facts across runs using a simple key–value store.
  
  ---
- ## 🎯 Why Memory?
  
  Without memory, an agent forgets everything once the script ends.  
  With memory, your agent can **persist preferences, facts, and context** across runs.  
  For example:
- Remember: *"I like intuitive explanations before math"*
- Recall: *"What style of explanation do I prefer?"*
- Apply: *"Now explain gradient descent in that style."*
  
  ---
- ## 📂 Project Structure
  
  ```text
  my-agent-project/
  │
  ├── agent_with_memory.py
  └── tools/
    └── memory_tools.py
- ---
- Install:
	- `$ pip install orjson`
-
- Code:
- agent_with_memory.py
  collapsed:: true
	- ```
	  from strands import Agent
	  from strands.models.ollama import OllamaModel
	  from tools.memory_tools import remember, recall, forget
	  
	  
	  model = OllamaModel(
	      host="http://localhost:11434",
	      model_id="llama3.1"   # make sure this is tool-capable
	  )
	  
	  agent = Agent(
	      model=model,
	      tools=[remember, recall, forget],
	      system_prompt=(
	          "You are my ML coach. "
	          "Before answering, check if any preferences are stored via recall."
	      ),
	  )
	  
	  if __name__ == "__main__":
	      # Store a preference
	      print(agent("Use the remember tool to save that I like 'intuitive before math'."))
	  
	      # Retrieve it
	      print(agent("What did I say about explanation style? Use recall."))
	  
	      # Apply it
	      print(agent("Explain gradient descent to me in my preferred style."))
	  
	  ```
- tools/memory_tools.py
	- ```
	  from pathlib import Path
	  import orjson
	  from typing import Optional
	  from strands import tool
	  
	  DB_PATH = Path("./memory_kv.json")
	  
	  def _load_db() -> dict:
	      if DB_PATH.exists():
	          return orjson.loads(DB_PATH.read_bytes())
	      return {}
	  
	  def _save_db(db: dict) -> None:
	      DB_PATH.write_bytes(orjson.dumps(db, option=orjson.OPT_INDENT_2))
	  
	  @tool
	  def remember(key: str, value: str) -> str:
	      """Store a fact under a key. Example: remember('style','explain simply')"""
	      db = _load_db()
	      db[key] = value
	      _save_db(db)
	      return f"Noted `{key}`."
	  
	  @tool
	  def recall(key: str) -> Optional[str]:
	      """Get a previously stored fact by key."""
	      db = _load_db()
	      return db.get(key)
	  
	  @tool
	  def forget(key: str) -> str:
	      """Delete a stored fact."""
	      db = _load_db()
	      if key in db:
	          del db[key]
	          _save_db(db)
	          return "Forgot."
	      return "Nothing to forget."
	  ```
-
- Now can execute python! and going to have conext!
-
- ---
- You can reachout me over linkedin https://www.linkedin.com/in/keerthivasan-kannan/
  
  See you on next article!
-