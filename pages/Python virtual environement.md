- #how_to
- > Article written by AI but facts are valid
-
- ## 💡 What is a Virtual Environment?
- A **virtual environment** is like a sandbox for your Python project.
- It keeps project dependencies **isolated** so different projects don’t mess with each other.
- Each environment can have its own Python packages (and versions).
  
  Think of it as having a **separate workspace** for every project.
  
  ---
- ## 🚀 Why Use Virtual Environments?
- Avoid version conflicts between projects.
- Keep your global Python clean.
- Reproduce the same setup easily across machines.
- Make collaboration smoother (especially with `requirements.txt`).
  
  ---
- ## 🛠️ Setting Up a Virtual Environment
- ### Step 1: Check Python version
  ```bash
  python3 --version
  ```
-
- ### Step 2: Create a virtual environment
  ```bash
  python3 -m venv myenv
  ```
  > Replace `myenv` with your preferred folder name
- ### Step 3: Activate the environment
- **Linux / Mac**
  ```bash
  source myenv/bin/activate
  ```
  **Windows (PowerShell)**
  ```powershell
  .\myenv\Scripts\activate
  ```
- > Once activated, your terminal will show something like: `(myenv)`
- ### Step 4: Install packages inside venv
  ```bash
  pip install requests
  ```
- ### Step 5: Deactivate
  ```bash
  deactivate
  ```
  
  ---
- ## 📦 Saving and Sharing Dependencies
- ### Freeze installed packages
  ```bash
  pip freeze > requirements.txt
  ```
-
- ### Install from file (for teammates)
  ```bash
  pip install -r requirements.txt
  ```
  
  ---