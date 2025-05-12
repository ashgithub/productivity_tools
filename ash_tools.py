#!/usr/bin/env -S /opt/homebrew/bin/uv  run 
##/usr/bin/env -S /opt/homebrew/bin/uv  run --script


# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "envyaml>=1.10.211231",
#    "langchain>=0.3.25",
#    "langchain-community==0.3.19",
#    "langchain-google-genai>=2.1.4",
#    "langchain-ollama>=0.3.2",
#    "oci>=2.150.3",
# ]
# ///
import sys,os 
import requests,json
import argparse, re
from pathlib import Path
from typing import Dict, Any
from envyaml import EnvYAML
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model_name="n/a"

def load_config(config_path: str = "ashtools_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable support."""
    try:
        return dict(EnvYAML(config_path))
    except FileNotFoundError:
        print(f"Config file not found: {config_path}. Using default configuration.")
        return {"model": {"type": "ollama"}}

# Model factory functions
def create_oci_model(params: Dict[str, Any]):
    from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
    return ChatOCIGenAI(
        model_id=params.get("model_name", "cohere.command-r-08-2024"),
        service_endpoint=params.get("endpoint", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"),
        compartment_id=params.get("compartment_id"),
        model_kwargs=params.get("model_kwargs", {"temperature": 0.7, "max_tokens": 500}),
        auth_profile=params.get("auth_profile", "DEFAULT")
    )

def create_gemini_model(params: Dict[str, Any]):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=params.get("model_name", "gemini-2.0-flash"),
        temperature=params.get("temperature", 0.7),
        max_output_tokens=params.get("max_tokens", 500),
        google_api_key=params.get("api_key")
    )

def create_ollama_model(params: Dict[str, Any]):
    from langchain_ollama import ChatOllama
    headers={"Authorization": f"Basic {params.get('api_key')}"}
    return ChatOllama(
        model=params.get("model_name", "llama3-groq-tool-use"),
        base_url=params.get("url", "https://macmini.industrylab.uk/ollama/"),
        temperature=params.get("temperature", 0.7),
        client_kwargs={"headers":headers}
    )

def get_chat_model(config: Dict[str, Any]):
    """Initialize and return the appropriate chat model based on configuration."""
    # Dictionary mapping model types to their factory functions
    model_factories = {
        "oci": create_oci_model,
        "gemini": create_gemini_model,
        "ollama": create_ollama_model
    }
    
    model_type = config.get("model", {}).get("type", "ollama")
    model_params = config["model"].get(f"{model_type}_params", {})
    
    # Get the appropriate factory function or default to ollama
    factory_func = model_factories.get(model_type, create_ollama_model)
    
    global model_name
    model_name = model_params.get('model_name')
    # Create and return the model
    return factory_func(model_params)

# for proof 
def extract_text_between_tags(text):
    pattern = r"{{(.*?)}}"
    matches = re.findall(pattern, text)
    return matches

# for proof 
def call_model(human, system, ai,chat):
    messages = [
        SystemMessage(content=system),
        AIMessage(content=ai),
        HumanMessage(content=human),
    ]
    response = chat.invoke(messages)
    return response.content


def main(args):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process input with different prompt modes")
    parser.add_argument("-m","--mode",  choices=["explain", "cmd", "help", "lookup", "proof"], required=True,
                        help="The type of prompt to use (default: explain)")
    parser.add_argument("-c", "--config_path", nargs="?", type=Path, default=Path("/Users/ashish/work/code/python/ash_tools/ashtools_config.yaml"),
                        help="Path to configuration file")
    parser.add_argument("input_text", nargs=argparse.REMAINDER, default=None,
                        help="Text input to process; if omitted, will read from stdin")
    
    
    # Parse known args (to extract mode and config_path)
    args, unknown = parser.parse_known_args(args)
    
    # Read input from command-line or stdin
    if args.input_text:
        input_text = ' '.join(args.input_text)
    else:
        input_text = sys.stdin.read()
   
    try:
        response = requests.get('https://macmini.industrylab.uk', timeout=1)  # Wait for 5 seconds
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.Timeout as e:
        os.environ["http_proxy"] = 'www-proxy-ash7.us.oracle.com:80'
        os.environ["https_proxy"] = 'www-proxy-ash7.us.oracle.com:80'
    except requests.exceptions.RequestException as e:
        pass 

    # Load configuration
    config = load_config(args.config_path)
    
    # Initialize the appropriate chat model
    chat = get_chat_model(config)
   
    return chat, config, input_text

def explain():
    chat,config,input_text = main(["-m","explain"]+sys.argv[1:])
    
    response = call_model(input_text,config.get("explain_prompt"), "I'm ready to  explain your input.",chat)
    print(f"[{model_name}]\n{response}")

def help():
    chat,config,input_text = main(["-m","help"]+sys.argv[1:])
    response = call_model(input_text,config.get("help_prompt"), "I'm ready to suggest commands for your input.",chat)


    print(f"[{model_name}]\n{response}")

def lookup():
    chat,config,input_text = main(["-m","lookup"]+sys.argv[1:])
    response = call_model(input_text,config.get("lookup_prompt"), "I'm ready to answer your question.",chat)
    print(f"[{model_name}]\n{response}")

def cmd():
    chat,config,input_text = main(["-m","cmd"]+sys.argv[1:])
    response = call_model(input_text,config.get("cmd_prompt"), "I'm ready to suggest commands for your input.",chat)
    try:
        alternatives = json.loads(response)
        for alt,cmd in alternatives.items():
            print (cmd)
    except Exception  as e:
        print(f"error: {e}\n{response}")
        

def proof():
    chat,config,input_text = main(["-m","proof"]+sys.argv[1:])
    #extract tags between input
    final_text = input_text
    prompt_text = input_text
    
    extracted_text = extract_text_between_tags(input_text)
    
    for tag in extracted_text:
        response = call_model(tag,config.get("proof_expand_prompt"),"I am ready to expand tags fro your template",chat)
        final_text = final_text.replace("{{"+tag+"}}",response)
        prompt_text = prompt_text.replace("{{"+tag+"}}","")

    if  prompt_text and prompt_text.isspace() == False:
        final_text = call_model(final_text, config.get("proof_clean_prompt"),"I am ready to proof read your text and edit it",chat)
    
    print(final_text, end='', flush=True)

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        main(sys.argv[1:])
    else: 
        try: 
            func_name = sys.argv[2]
            # Check if the function exists in the global namespace and call it
            if func_name in globals() and callable(globals()[func_name]):
                globals()[func_name]()
            else:
                print(f"Unknown function: {func_name}")
        except Exception as e:
            print(f"error: {e}")