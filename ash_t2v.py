import argparse
import os,io,time,json
from bs4 import BeautifulSoup
import requests
from envyaml import EnvYAML
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import *
from oci.config import from_file
from oci.object_storage import ObjectStorageClient


# Load configuration from YAML file
def load_config(config_path: str = "asht2v_config.yaml") -> Dict[str, Any]:
	"""Load configuration from YAML file with environment variable support."""
	try:
		return dict(EnvYAML(config_path))
	except FileNotFoundError:
		print(f"Config file not found: {config_path}. Using default configuration.")
		exit

import os
import uuid
import hashlib
import shutil

oci_config = None

def process_file_uri(config,file_path,  video_name):
	"""
	Process a file URI by copying it to the cache directory and checking for existing transcripts.

	Args:
		config (dict) : config file
		file_path (str): Path to the file
		video_name (str, optional): Optional video name. Defaults to None.

	Returns:
		str: Path to the cached video file
		str: Path to the transcript file
	"""
	cache_dir = os.path.expanduser(config["cache_dir"])
	print(f"using {cache_dir=}")
	# Calculate the hash of the file contents
	file_hash = calculate_file_hash(file_path)

	# Check if a file with the same hash exists in the cache
	cached_video_dir, cached_video_path = find_cached_video(cache_dir, file_hash)
	
	if cached_video_dir:
		print(f"Video already exists in cache: {cached_video_path} (found in directory: {cached_video_dir})")

	if cached_video_path is None:
		# Create a subfolder under the cache directory with the video name
		print(f"video not found, creating new folder {video_name=} in {cache_dir=}")
		cached_video_dir = os.path.join(cache_dir, video_name)
		os.makedirs(cached_video_dir, exist_ok=True)
		
		# Copy the file to the cache directory
		cached_video_path = os.path.join(cached_video_dir, os.path.basename(file_path))
		shutil.copyfile(file_path, cached_video_path)
		print(f"video copied to {cached_video_path=}")

	# Check if a transcript exists for the cached video
	transcript_path = find_transcript(cached_video_dir, cached_video_path)
	if transcript_path:
		print(f"Transcript already exists: {transcript_path}")
	else:
		# Upload file to object bucket and transcribe using OCI Speech Whisper model
		upload_video(config["bucket"],cached_video_path)
		result = transcibe(config["oci"], config["oci_speech_config"],config["bucket"],os.path.basename(file_path))  
		transcript_path = download_transcript(config["bucket"],cached_video_dir,result.output_location.prefix) 

	return transcript_path


def calculate_file_hash(file_path):
	"""
	Calculate the SHA-256 hash of a file.

	Args:
		file_path (str): Path to the file

	Returns:
		str: Hexadecimal representation of the file hash
	"""
	with open(file_path, "rb") as f:
		file_hash = hashlib.sha256(f.read()).hexdigest()
	return file_hash


def find_cached_video(cache_dir, file_hash):
	"""
	Find a cached video with the matching hash.

	Args:
		cache_dir (str): Cache directory path
		file_hash (str): Hash of the file contents

	Returns:
		tuple: (directory_path, file_path) if found, otherwise (None, None)
	"""
	for root, dirs, files in os.walk(cache_dir):
		for file in files:
			file_path = os.path.join(root, file)
			if calculate_file_hash(file_path) == file_hash:
				return root, file_path
	return None, None


def find_transcript(cache_dir, video_path):
	"""
	Find a transcript for a given video.

	Args:
		cache_dir (str): Cache directory path
		video_path (str): Path to the video file

	Returns:
		str: Path to the transcript file, or None if not found
	"""
	transcript_path = video_path + ".transcript.txt"
	if os.path.exists(transcript_path):
		return transcript_path
	return None
def upload_video(bucket_cfg, file):
	object_storage_client = ObjectStorageClient(oci_config)
	print(f"Uploading file {file} ...")
	object_storage_client.put_object(bucket_cfg["namespace"], 
									 bucket_cfg["bucket_name"], 
									 f"{bucket_cfg['prefix']}/{os.path.basename(file)}", 
									 io.open(file,'rb'))
	print("Upload completed !")
	
def download_transcript(bucket_cfg,dir,output_prefix):
	object_storage_client = ObjectStorageClient(oci_config)
	list_objects_response = object_storage_client.list_objects(
									bucket_cfg["namespace"], 
									bucket_cfg["bucket_name"],
									prefix=output_prefix)
	for obj in list_objects_response.data.objects:
		response  = object_storage_client.get_object(bucket_cfg["namespace"], bucket_cfg["bucket_name"], obj.name)
		filename = os.path.join(dir, f"{os.path.splitext(os.path.basename(obj.name))[0]}.transcript.txt")
		
		transcript_text = json.loads(response.data.text)["transcriptions"][0]["transcription"]
		
		with open(filename,"w") as f:
			f.write(transcript_text)
		print (f"saved {filename}") 
		return filename
def transcibe(oci_cfg,speech_cfg,bucket_cfg, file_name):
	speech_client = AIServiceSpeechClient(config=oci_config, service_endpoint=speech_cfg["service_endpoint"] )

	#create transcription job
	bucket_video_location = ObjectLocation(namespace_name=bucket_cfg["namespace"], bucket_name=bucket_cfg["bucket_name"],
																	object_names=[f"{bucket_cfg['prefix']}/{file_name}"])
	bucket_transcript_location = OutputLocation(namespace_name=bucket_cfg["namespace"], bucket_name=bucket_cfg["bucket_name"],
																	prefix=bucket_cfg['prefix'])
	input_location = ObjectListInlineInputLocation(location_type="OBJECT_LIST_INLINE_INPUT_LOCATION", object_locations=[bucket_video_location])
	model = TranscriptionModelDetails(language_code="en", model_type="WHISPER_MEDIUM", domain="GENERIC", transcription_settings = TranscriptionSettings(diarization=Diarization(is_diarization_enabled=False)))
	
	transcription_job_details = CreateTranscriptionJobDetails(display_name=f"talk-2-video",model_details=model, normalization=TranscriptionNormalization(is_punctuation_enabled=True),
																compartment_id=oci_cfg["compartment_id"],
																input_location=input_location, output_location=bucket_transcript_location)
	transcription_job = speech_client.create_transcription_job(create_transcription_job_details=transcription_job_details)
	transcribe_job_id = transcription_job.data.id
	print(f"Transcription Job ID: {transcribe_job_id}")
	print("polling. The transcription can take 50% of your video length")

	#poll for job to complete 
	result = None
	while True:
		transcription_job = speech_client.get_transcription_job(transcribe_job_id)
		job_status = transcription_job.data.lifecycle_state
		print(f"Current Status: {job_status}", end='\r')
	
		if job_status == "SUCCEEDED":
			print("\nTranscription job completed successfully!")
			result = transcription_job.data        
			break
		elif job_status == "FAILED":
			print("\nTranscription job failed.")
			exit
		else:
			time.sleep(5)  # Wait 5 seconds before checking again 

	return result

def process_webpage_uri(uri):
	# Extract video links using Beautiful Soup
	response = requests.get(uri)
	soup = BeautifulSoup(response.content, 'html.parser')
	video_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.mp4')]

	# Prompt user to choose a link if multiple options exist
	if len(video_links) > 1:
		print("Multiple video links found:")
		for i, link in enumerate(video_links):
			print(f"{i+1}. {link}")
		choice = int(input("Enter the number of the desired video: "))
		chosen_link = video_links[choice - 1]
	else:
		chosen_link = video_links[0]

	# Download chosen video
	response = requests.get(chosen_link)
	with open(os.path.join(config['cache_dir'], os.path.basename(chosen_link)), 'wb') as f:
		f.write(response.content)

	# Check if transcript exists in cache
	transcript_path = os.path.join(config['cache_dir'], f"{os.path.basename(chosen_link)}.transcript")
	if os.path.exists(transcript_path):
		print("Transcript already exists in cache")
	else:
		# Upload video to object bucket and transcribe using OCI Speech Whisper model
		obj_storage_client = OCIObjectStorageClient(config['oci_config'])
		obj_storage_client.upload_file(os.path.join(config['cache_dir'], os.path.basename(chosen_link)), config['bucket_name'])
		speech_client = OCISpeechClient(config['oci_speech_config'])
		transcript = speech_client.transcribe_file(obj_storage_client.get_object_url(config['bucket_name'], os.path.basename(chosen_link)))
		with open(transcript_path, 'w') as f:
			f.write(transcript)

def create_ollama_model(params: Dict[str, Any]):
	headers={"Authorization": f"Basic {params.get('api_key')}"}
	return ChatOllama(
		model=params.get("model_name", "llama3-groq-tool-use"),
		base_url=params.get("url", "https://macmini.industrylab.uk/ollama/"),
		temperature=params.get("temperature", 0.7),
		client_kwargs={"headers":headers}
	)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('uri', help='URI pointing to a webpage or file containing a video')
	parser.add_argument('--name', help='Optional video name')
	parser.add_argument('-c','--ctx_file', help='Optional context file describing the video')
	args = parser.parse_args()

	config = load_config("asht2v_config.yaml")
	
	global oci_config
	oci_config= from_file( os.path.expanduser(config["oci"]["config"]), config["oci"]["profile"])
	
	video_name = args.name
	if video_name is None:
		video_name = os.path.splitext(os.path.basename(args.uri))[0] + "_" + str(uuid.uuid4())[:8]
		print(f"using the name {video_name}")
	if args.uri.startswith('http'):
		transcript_path = process_webpage_uri(config, args.uri, args.name)
	else:
		transcript_path = process_file_uri(config,args.uri, args.name)

	llm = create_ollama_model(config["llm_ollama"])
	transcript = ""
	context = ""
	if args.ctx_file:
		with open(args.ctx_file, 'r') as file:
			context = file.read()
	with open(transcript_path, 'r') as file:
		transcript = file.read()

	messages = [
		SystemMessage(content=config["llm_ollama"]["summarize_prompt"]),
		AIMessage(content=f"I'm ready to summarize your meeting and then answer questions. paste the transcript"),
		HumanMessage(content=f"{context} {transcript}"),
	] 
	max_history = config["llm_ollama"]["history"]
	# Enter Q&A loop
	while True:
		answer = llm.invoke(messages)
		print(answer.content)
		messages.append(AIMessage(content=answer.content))
		question = input("Ask a question: ")
		messages.append(HumanMessage(content=question))
		if len(messages) > max_history:
			messages.pop(3)

if __name__ == '__main__':
	main()