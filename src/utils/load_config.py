import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from langchain_groq import ChatGroq
#import langchain_ollama as ChatOllama  # Uncomment if needed
import chromadb

# Load environment variables
load_dotenv()
print("Environment variables are loaded.")

class LoadConfig:
    def __init__(self) -> None:
        app_config = self._load_app_config()
        
        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_chroma_client()
        self.load_groq_client()  # Initialize Groq client
        self.load_rag_config(app_config=app_config)

        # Uncomment the code below if you want to clean up the upload csv SQL DB on every fresh run of the chatbot (if it exists)
        # self.remove_directory(self.uploaded_files_sqldb_directory)

    def _load_app_config(self):
        """Load application configuration from YAML file."""
        with open(here("configs/app_config.yml")) as cfg:
            return yaml.safe_load(cfg)  # Use safe_load for security

    def load_directories(self, app_config):
        """Load directory paths from the configuration."""
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"]
        )
        self.sqldb_directory = str(
            here(app_config["directories"]["sqldb_directory"])
        )
        self.uploaded_files_sqldb_directory = str(
            here(app_config["directories"]["uploaded_files_sqldb_directory"])
        )
        self.stored_csv_xlsx_sqldb_directory = str(
            here(app_config["directories"]["stored_csv_xlsx_sqldb_directory"])
        )
        self.persist_directory = app_config["directories"]["persist_directory"]

    def load_llm_configs(self, app_config):
        """Load language model configurations."""
        self.model_name = os.getenv("GROQ_MODEL_NAME", "llama3-groq-70b-8192-tool-use-preview")
        self.agent_llm_system_role = app_config["llm_config"]["agent_llm_system_role"]
        self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.embedding_model_name = os.getenv("embed_deployment_name", "sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the Llama model using ChatGroq
        self.langchain_llm = ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature
        )

    def load_groq_client(self):
        """Initialize the Groq client."""
        self.groq_client = ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature
        )

    def load_chroma_client(self):
        """Initialize the Chroma client."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(here(self.persist_directory))
        )

    def load_rag_config(self, app_config):
        """Load RAG configurations."""
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed."
                )
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")