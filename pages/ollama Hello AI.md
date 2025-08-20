- #agentic_ai
-
- [[Python virtual environment]] We have seen how to create virtual python environment to isolate the environment
-
- Install ollama on your system. Once installed get it running on terminal
	- Also install python ollama package
		- `$ pip install ollama`
	- `$ ollama serve`
-
- To check what packages installed in ollama
	- `$ ollama list`
-
- Install specific model - We are using llama3 which is of around 4GB size for our setup
	- `$ ollama pull llama3`
-
- Hello World Python file
- > As strands is opensource tool from AWS it's going to take Bedrock as default if no model is specified. In our case we explicitly mention ollama as our model
- ```
  from strands import Agent
  from strands.models.ollama import OllamaModel
  
  model = OllamaModel(
      host="http://localhost:11434",  # default Ollama port
      model_id="llama3"               # or any pulled model like mistral
  )
  
  agent = Agent(model=model)
  response = agent("What is agentic AI?")
  print(response)
  ```
-
- Time to run our code!
- `$python3 hello-ai.py`
-
- ---
- You can connect me over linkedin: https://www.linkedin.com/in/keerthivasan-kannan/
-
- See you in next article :)
-