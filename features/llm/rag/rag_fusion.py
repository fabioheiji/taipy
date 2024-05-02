import re
from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Rag_fusion():
    """
    RAG Fusion class that defines the RAG Fusion pipeline
    """
    def __init__(self) -> None:
        self._llm_1 = None
        self._llm_2 = None
        self._question = None
        self._generate_queries = None
        self._retriever = None
        self._answer = None
    
    def run(self, llm_1, llm_2, question, retriever):
        self._llm_1 = llm_1
        self._llm_2 = llm_2
        self._retriever = retriever
        self._question = question
        self._create_generate_queries()
        self._run_invoke()
    
    def get_answer(self):
        return self._answer

    def _remove_empty_lines(self, docs):
        final_docs = []
        for doc in docs:
            # Remove empty lines
            if not re.search(r'^\s*$',str(doc)):
                final_docs.append(doc)
        return final_docs

    def _create_generate_queries(self):

        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        print(prompt_rag_fusion)
        print(self._llm_1)

        self._generate_queries = (
            prompt_rag_fusion 
            | self._llm_1
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
            | self._remove_empty_lines
        )

    def _reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results

    def _run_invoke(self):

        retrieval_chain_rag_fusion = self._generate_queries | self._retriever.map() | self._reciprocal_rank_fusion

        # RAG
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            prompt
            | self._llm_2
            | StrOutputParser()
        )

        self._answer = final_rag_chain.invoke({"context": retrieval_chain_rag_fusion,"question":self._question})