# Chat PDF Assistant
This project enables a local chat interface for interacting with PDF documents.
The code implements a Streamlit application that allows users to interact with a chatbot capable of understanding and answering questions based on the content of uploaded PDF documents. It leverages the LangChain Community libraries for processing and querying document contents using a vector store and a local LLM Ollama Mistral. The application is designed for local use, providing a privacy-focused way to analyze and extract information from documents.
<img width="1312" alt="image" src="https://github.com/i038615/local_rag/assets/44123175/19d2c328-725f-4e74-b6f6-2ba07feb341f">

## Dependencies
- **LangChain Community**: For vector storage, chat models, embeddings, and document loading.
- **Streamlit**: For creating the web application interface.
- **Ollama Mistral**: The underlying large language model used for generating answers.

## Installation
To set up your environment to run this code, follow these steps:
1. **Clone the repository**:
```bash
git clone https://github.com/i038615/local_rag/
cd [repository-directory]
```
2. **Install dependencies**:
Ensure you have Python 3.7+ installed, then run:
```bash
pip install -r requirements.txt
```
3. **Model Setup**:
This project uses the Ollama Mistral model. Follow the official Ollama documentation to install and set up the Mistral model for local usage.

## Usage
To start the ollama server, run:
```bash
ollama serve
```
To start the application, run:
```bash
streamlit run local_chat.py
```

## How it works
__ChatPDFAssistant Class__: 
This is the core class that handles the PDF ingestion, text extraction, and setup of the chat model pipeline. 
It consists of several key methods:

____init____: Initializes the ChatOllama model, sets up text splitting for large documents, and prepares the prompt template used for querying the model.

___create_prompt_template__:Defines the template for how questions are presented to the chat model, emphasizing that answers should be based directly on the provided context.

__ingest_pdf__: Loads a PDF document, splits it into manageable chunks, and indexes these chunks in a vector store for efficient retrieval.

___prepare_retriever__: Sets up the retrieval mechanism that allows querying the vector store to find relevant document chunks based on a similarity score.

__ask__: Processes a query by retrieving relevant context from the vector store and passing it along with the query to the chat model for generating an answer.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
