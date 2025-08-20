- #agentic_ai
-
- After exploring [[Python virtual environment]] and [[ollama Hello AI]] To we will explore integrating tools on strands
-
- Install Tools
	- `$ pip install "strands-agents[ollama]" strands-agents-tools`
-
- > Initial undertstanding: Tool is more writing custom functions and letting llm to choose when to utilise them
-
- Sample Code:
	- ```
	  from strands import Agent, tool
	  from strands.models.ollama import OllamaModel
	  
	  @tool
	  def add(a: int, b: int) -> int:
	      "Add two integers."
	      return a + b
	  
	  @tool
	  def multiply(a: int, b: int) -> int:
	      "Multiply two integers."
	      return a * b
	  
	  agent = Agent(
	  model=OllamaModel(
	           host="http://localhost:11434",
	           model_id="llama3.1"   # or "qwen2.5" / "qwen3"
	      ),
	      tools=[add,multiply],
	      system_prompt="Use tools when helpful."
	  )
	  
	  print(agent("Multiply 7 and 5 using the tool."))
	  
	  print(agent("Multiply 12 by 9 using the tool."))
	  
	  
	  ```
-
- ---
- Stay tuned for more experimentation on AI : Connect me over linkedin: https://www.linkedin.com/in/keerthivasan-kannan/
-
- Stay tuned! Cheers!