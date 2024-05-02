from taipy.gui import Markdown
from features.llm.llm import LLM_prompt
from features.llm.run_llm_module import run_llm_module
import os
from pathlib import Path
from time import sleep
import markdown


question = ""
answer = ""
llm_md = Markdown(
"""

# Module *Large Language Model*



<|{question}|input|multiline|lines_shown=3|label=Type here you question|class_name=full_width|on_change=save_question|>


<|Ask for AI|button|on_action=on_click_ask_ai|>

<|{answer}|text|>
"""
)

def save_question(state):
    print(state.question)

def on_click_ask_ai(state):

    question_txt = state.question
    print('\n\n\n')
    print(f'The question is: {question_txt}')
    print('\n\n\n')
    # state.answer = "The answer is: 42"

    # llm_prompt = LLM_prompt()
    # model_path = os.path.join(Path(os.getcwd()), "features", "llm", "models", 'Meta-Llama-3-8B-Instruct.Q8_0.gguf')
    # llm_1 = llm_prompt.ll_model(model_path=model_path, stop_sequences = ["```"])
    
    # llm_2 = llm_prompt.ll_model(model_path=model_path)
    # path_file = os.path.join(Path(os.getcwd()), "data", "llm", "Relating Graph Neural Networks to Structural Causal Models.pdf")
    # llm_prompt.create_retriever(path_file=path_file)

    # answer_llm = llm_prompt.call_rag(rag_name='rag_fusion', llm_1=llm_1, llm_2=llm_2, question=question_txt)
    answer_llm = run_llm_module(question_txt)
    try:
        answer_llm = answer_llm.split("output_answer = ")[1].split('return output_answer')[0].strip()
    except Exception as e:
        print(e)
        pass
    state.answer = answer_llm
    print('\n\n\nFAAAAAAABIIIIIIIOOOOOOO')
    print(answer_llm)    
    print('nFAAAAAAABIIIIIIIOOOOOOO\n\n\n')
