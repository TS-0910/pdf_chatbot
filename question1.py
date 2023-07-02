import os, openai, sys, re
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate

#abstractのみ抽出する関数
def abstract_extraction(pdf_file_path):    
    #PDF読み込み
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    
    #Abstractから1 Introductionまでを抜き出す
    pattern = r"Abstract(.*?)1 Introduction"
    match = re.search(pattern, docs[0].page_content, re.DOTALL | re.IGNORECASE)

    #もしマッチすれば、docsをそれだけにする
    if match:
        docs[0].page_content = match.group(1)
        docs = docs[0]
        return [docs]
    #マッチしない時は2ページ目のInstructionまでを持ってくる
    else :
        pattern_2ndpage = r"(.*?)1 Introduction"
        match_2ndpage = re.search(pattern_2ndpage, docs[1].page_content, re.DOTALL | re.IGNORECASE)
        if match_2ndpage:
            docs[1].page_content = match_2ndpage.group(1)
            docs = docs[:2]
            return docs
        #それでもマッチしない時は2ページ目までを持ってくる
        else : 
            docs = docs[:2]
            return docs

def summarize_pdf(pdf_file_path, llm):
    #abstractだけ持ってくる
    docs = abstract_extraction(pdf_file_path)
    prompt_template = """    
    Write a concise summary of this in one sentence. Start with 'This is a paper about'. Traslate into Japanese at the end.        
    
    {text}

    Summary in Japanese: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                map_prompt=PROMPT, combine_prompt=PROMPT)
    custom_summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
    
    #1文にする
    custom_summary = custom_summary.split("。")[0] + "。"
    
    return custom_summary


def main():
    url = sys.argv[1]    
    openai_key = sys.argv[2]
    os.environ["OPENAI_API_KEY"] = f"{openai_key}"
    
    #llm指定
    llm = OpenAI(temperature=0)

    return summarize_pdf(f"{url}", llm)
    
if __name__ == "__main__":
    summary = main()
    print(summary)