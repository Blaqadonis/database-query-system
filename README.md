# KnowledgeBlaq


```KnowledgeBlaq``` is a multi-agent AI application that enables users to interact with SQL databases and using Retrieval-Augmented Generation (RAG) to respond to queries. It supports multiple modes of operation including querying stored SQL databases, processing CSV/XLSX data, and performing RAG-based searches with embeddings from ChromaDB. The chatbot leverages LangChain agents, Hugging Face embeddings, and can now work with both SQLite (for local development) and PostgreSQL (for production).


**Key NOTE:** Remember to NOT use a SQL database with ```WRITE``` privileges. Use only ```READ``` and limit the scope. Otherwise your user could manupulate the data (e.g ask your chain to delete data).

## **Table of Contents**
* Features
* Technologies
* Installation
* Dependencies
* Setting Up PostgreSQL
* Configuration
* Environment Variables
* config.yaml
* Usage
* PostgreSQL Setup


### **Features**
* ```SQL Database Querying```: The chatbot can query data stored in SQL databases (both SQLite and PostgreSQL).
* ```Embeddings-Based RAG```: Uses Hugging Face models to generate document embeddings and retrieve relevant information from ChromaDB.
* ```Multi-File Upload Support```: Supports uploading CSV and XLSX files, which are converted to a SQLite/PostgreSQL database for querying.
* ```Flexible Configuration```: Easily configurable through a config.yaml file and environment variables.
### **Technologies**
* Python 3.10
* LangChain: For managing LLM chains and query agents.
* Hugging Face: For generating embeddings.
* SQLAlchemy: For interacting with SQL databases (SQLite and PostgreSQL).
* ChromaDB: For handling vector-based searches.
* PostgreSQL: For production-grade SQL database management.
* Gradio: For chatbot UI.
### **Installation**
#### Dependencies
To get started, first clone this repository and install the dependencies:

bash
Copy code
git clone https://github.com/your-repo-name.git
cd your-repo-name

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or for Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Setting Up PostgreSQL
Install PostgreSQL:

On Ubuntu:
bash
Copy code
sudo apt update
sudo apt install postgresql postgresql-contrib
On macOS (with Homebrew):
bash
Copy code
brew install postgresql
brew services start postgresql
On Windows: Download PostgreSQL and follow the installation instructions.
Create a PostgreSQL Database: Open the PostgreSQL shell or any PostgreSQL management tool (like pgAdmin), and create a new database:

sql
Copy code
CREATE DATABASE mydbname;
Create a PostgreSQL User: You can create a new user with a password:

sql
Copy code
CREATE USER myusername WITH PASSWORD 'mypassword';
Grant Permissions: Grant the user access to the database:

sql
Copy code
GRANT ALL PRIVILEGES ON DATABASE mydbname TO myusername;
Configuration
Environment Variables
You can manage environment variables through a .env file. Here is an example of the variables required for PostgreSQL and Hugging Face embeddings:

makefile
Copy code
POSTGRES_USER=myusername
POSTGRES_PASSWORD=mypassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mydbname

embed_deployment_name=sentence-transformers/all-MiniLM-L6-v2
GROQ_MODEL_NAME=llama3-groq-70b-8192-tool-use-preview
config.yaml
The chatbot configuration is controlled through a config.yaml file. Below is an example configuration:

yaml
Copy code
langchain_llm: groq  # Language model used for SQL queries
rag_llm_system_role: "You are a helpful assistant."

# ChromaDB settings
chroma_client:
  chromadb_host: "localhost"
  chromadb_port: 6333

collection_name: "my_collection"
top_k: 5  # Number of results to retrieve for RAG mode

# Database directories for SQLite (can be ignored if using PostgreSQL)
sqldb_directory: "data/sqldb.db"
uploaded_files_sqldb_directory: "data/uploaded_files.db"
stored_csv_xlsx_sqldb_directory: "data/stored_files.db"
Usage
To start the chatbot, you need to decide which mode you'll run it in. If you plan to use the PostgreSQL database, follow the instructions below.

Running the Chatbot
SQL Query Mode: The chatbot interacts with either SQLite or PostgreSQL databases to answer queries. It can execute complex SQL queries and return results directly.

RAG Mode: The chatbot can use embeddings from documents to perform Retrieval-Augmented Generation (RAG) with ChromaDB and answer the user's questions based on relevant document search results.

bash
Copy code
python src/app.py
This command will launch the chatbot, and you can interact with it using a Gradio interface.

PostgreSQL Setup
To switch from SQLite to PostgreSQL for a production environment:

Install PostgreSQL dependencies: If not already done, install the psycopg2 package to interact with PostgreSQL:

bash
Copy code
pip install psycopg2
Update Configuration: Modify the database URI in your code to use PostgreSQL instead of SQLite. Update your config.yaml or environment variables to point to PostgreSQL:

python
Copy code
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

# PostgreSQL connection
engine = create_engine("postgresql://myusername:mypassword@localhost/mydbname")
db = SQLDatabase(engine=engine)
Run the Application: Once the PostgreSQL database is configured and accessible, your chatbot will be able to query the database and return results as it did with SQLite.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contributing
If you'd like to contribute to this project, feel free to submit a pull request or open an issue.