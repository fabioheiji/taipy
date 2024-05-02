import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain.llms import LlamaCpp, llamacpp #This class can interact with Llama 2 models using C++ bindings
from langchain.callbacks.manager import CallbackManager #This class can manage callbacks for various events during the execution of the code
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler #This class can print messages to the standard output stream during the execution of the code
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms import LlamaCpp

from features.llm.rag import rag_fusion


import torch


import numpy as np
import re

from pprint import pprint

class LLM_prompt:
    """
    This class provides a template for creating a prompt for a llm model using some RAG.
    """
    def __init__(self) -> None:
        pass

    def ll_model(
            self,
            n_gpu_layers: int = 30,
            n_batch: int = 512,
            model_path: str = os.path.join(os.getcwd(), 'models', 'Meta-Llama-3-8B-Instruct.Q8_0.gguf'),
            stop_sequences: list = [],
            verbose: bool = True,
            max_tokens: int = 512,
            n_ctx: int = 5000,
            temperature: float = 0,
            top_k: int = 1,
            repeat_penalty: float = 1.03
        ):

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) # Create an instance of CallbackManager with a list of callback handlers as an argument (in this case, only one handler that prints messages to stdout)

        # Create an instance of LlamaCpp
        return LlamaCpp(  
            model_path=model_path, # The path of the Llama 3 model file
            n_gpu_layers=n_gpu_layers, # The number of GPU layers to use for the model (this can affect the performance and memory usage) 
            n_batch=n_batch, # The number of batches to use for the model (this can affect the performance and memory usage)
            f16_kv=True,  # A flag to indicate whether to use 16-bit floating point numbers for key and value matrices (this can improve performance and memory usage). MUST set to True, otherwise you will run into problem after a couple of calls.
            callback_manager=callback_manager, # The callback manager that will handle the callbacks during the execution of the code (in this case, only one callback handler that prints messages to stdout)
            verbose=verbose, # A flag to indicate whether to print verbose messages
            max_tokens=max_tokens, # The maximum number of tokens that the model can generate
            n_ctx=n_ctx, # The maximum number of tokens that the model can process as context
            temperature=temperature, # The temperature parameter that controls how creative or diverse the model's output is (a lower temperature means more coherence and consistency)
            top_k=top_k, # The top-k parameter that controls how diverse the model's output is (a lower top-k means more coherence and consistency)
            # repetition_penalty=1.03, # The repetition penalty parameter that controls how repetitive the model's output is (a higher repetition penalty means less repetition)
            repeat_penalty=repeat_penalty,
            stop=stop_sequences,
        )

    def _define_device(self):
        if torch.cuda.is_available():
            # Use CUDA GPU
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            # Use Apple M1 Metal Acceleration
            return torch.device("mps")
        
        return torch.device("cpu")

    def _remove_empty_lines(self,docs):
        final_docs = []
        for doc in docs:
            # Remove empty lines
            if not re.search(r'^\s*$',str(doc)):
                final_docs.append(doc)
        return final_docs

    def _create_embedding_model(self,model_name = "BAAI/bge-small-en-v1.5"):
        device = self._define_device()
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,            
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def create_retriever(self,path_file, embedding_model_name="BAAI/bge-small-en-v1.5"):
        embedding_model = self._create_embedding_model(model_name=embedding_model_name)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, 
            chunk_overlap=50)
        loader = PyPDFLoader(path_file)
        documents = loader.load_and_split(text_splitter)
        vectorstore = Chroma.from_documents(documents=documents, 
                                            embedding=embedding_model)

        self.retriever = vectorstore.as_retriever()
    
    def call_rag(self, rag_name: Literal["rag_fusion"], llm_1, llm_2, question):
        if rag_name == "rag_fusion":
            rag = rag_fusion.Rag_fusion()
            rag.run(llm_1, llm_2, question, self.retriever)
            return rag.get_answer()

