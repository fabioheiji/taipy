import os
from pathlib import Path
import sys
# sys.path.append(str(Path(os.getcwd()).parent))
from pprint import pprint

from features.llm.llm import LLM_prompt

def run_llm_module(question: str):
    llm_prompt = LLM_prompt()
    model_path = os.path.join(Path(os.getcwd()), "features", "llm", "models", 'Meta-Llama-3-8B-Instruct.Q8_0.gguf')
    llm_1 = llm_prompt.ll_model(model_path=model_path, stop_sequences=["```"], max_tokens=1000)
    # llm_2 = llm_prompt.ll_model(model_path=model_path)
    path_file = os.path.join(Path(os.getcwd()), "data", "llm","Relating Graph Neural Networks to Structural Causal Models.pdf")
    llm_prompt.create_retriever(path_file=path_file)

    # question = """
    #     What are the key differences between Graph Neural Networks and Structural Causal Models?
    # """

    answer = llm_prompt.call_rag(rag_name='rag_fusion', llm_1=llm_1, llm_2=llm_1, question=question)

    print('\n\n\nFAAAAAAABIIIIIIIOOOOOOO')
    pprint(answer)
    print('nFAAAAAAABIIIIIIIOOOOOOO\n\n\n')
    return answer