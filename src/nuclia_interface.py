# src/nuclia_interface.py
"""
Handles all interactions with the Nuclia Knowledge Box (KB) API.

This module provides the NucliaKGHandler class, responsible for:
- Authenticating with a specific Nuclia Knowledge Box using an NUA key.
- Uploading documents to the Knowledge Box, which triggers Nuclia's
  internal processing to generate/update the Knowledge Graph.
- Placeholder methods for future querying of the Knowledge Graph once generated.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Official Nuclia Python SDK
from nuclia import sdk
from nuclia.sdk.kb import NucliaKB

# Project-specific configurations
from . import config


class NucliaKGHandler:
	"""
	Manages communication with a Nuclia Knowledge Box (KB).
	This class uses the Nuclia Python SDK for operations such as document
	upload (which in turn triggers Nuclia's graph generation) and,
	in the future, querying the generated Knowledge Graph.
	"""

	def __init__(self,
				 kb_url: Optional[str] = None,
				 api_key: Optional[str] = None):
		"""
		Initializes the NucliaKGHandler and authenticates with the specified Knowledge Box.

		The KB URL (which includes the KB ID) and the API Key (NUA key for that KB)
		are essential for targeting and authenticating with a specific Knowledge Box.

		Args:
			kb_url (Optional[str]): The full API URL for the Knowledge Box.
									e.g., "https://YOUR_REGION.nuclia.cloud/api/v1/kb/YOUR_KB_ID"
									If None, it's fetched from config.NUCLIA_KB_URL.
			api_key (Optional[str]): The NUA key for programmatic access to the KB.
									 If None, it's fetched from config.NUCLIA_API_KEY.
		
		Raises:
			ValueError: If essential kb_url or api_key is missing.
			ConnectionError: If authentication or initial connection to the KB fails.
			RuntimeError: For other unexpected SDK initialization errors.
		"""
		self.logger = logging.getLogger(__name__)

		resolved_kb_url: Optional[str] = kb_url or config.NUCLIA_KB_URL
		resolved_api_key: Optional[str] = api_key or config.NUCLIA_API_KEY

		if not resolved_kb_url:
			msg = "Nuclia KB URL is not configured. Cannot initialize NucliaKGHandler."
			self.logger.critical(msg)
			raise ValueError(msg)
		if not resolved_api_key:
			msg = "Nuclia API Key (NUA Key for KB) is not configured. Cannot initialize NucliaKGHandler."
			self.logger.critical(msg)
			raise ValueError(msg)

		self._kb_url: str = resolved_kb_url  # Now asserted to be str
		self._api_key: str = resolved_api_key # Now asserted to be str

		try:
			self.auth = sdk.NucliaAuth()
			# Authenticates and configures the SDK to target the specified KB.
			self.auth.kb(url=self._kb_url, token=self._api_key)
			
			self.kb: NucliaKB = sdk.NucliaKB() 
			
			self.logger.info(f"Successfully authenticated with Nuclia KB ID: '{self.kb.kbid}' at URL: {self.kb.url}")

		except Exception as e:
			self.logger.error(f"Unexpected error during NucliaKGHandler initialization: {e}", exc_info=True)
			raise RuntimeError(f"Unexpected error initializing NucliaKGHandler: {e}")

	def upload_documents_from_folder(
			self,
			folder_path: Path,
			max_files_to_upload: Optional[int] = None
		) -> Dict[str, Dict[str, Any]]:
		"""
		Uploads all .txt files from a local folder to the configured Nuclia KB.

		Each uploaded file becomes a resource in Nuclia. Nuclia will then asynchronously
		process these resources, which includes extracting text, entities, relationships,
		and ultimately generating/updating the Knowledge Graph for this KB.

		Args:
			folder_path (Path): Local directory path containing .txt files.
			max_files_to_upload (Optional[int]): Max files to upload. If None, uploads all.

		Returns:
			Dict[str, Dict[str, Any]]: Summary of upload status for each attempted file.
									   Format: {filename: {"status": str, "details": str, 
														  "resource_id": Optional[str], "slug": str}}
		
		Raises:
			FileNotFoundError: If folder_path doesn't exist or isn't a directory.
		"""
		self.logger.info(f"Starting document ingestion into Nuclia KB ID: '{self.kb.kbid}' from folder: {folder_path}")
		if not folder_path.is_dir():
			msg = f"Specified folder path for upload is not a directory or does not exist: {folder_path}"
			self.logger.error(msg)
			raise FileNotFoundError(msg)

		upload_results: Dict[str, Dict[str, Any]] = {}
		files_processed_count = 0

		for item_path in sorted(list(folder_path.iterdir())):
			if not (item_path.is_file() and item_path.name.lower().endswith(".txt")):
				continue # Skip non-.txt files or directories

			if max_files_to_upload is not None and files_processed_count >= max_files_to_upload:
				self.logger.info(f"Reached upload limit of {max_files_to_upload} files. Halting uploads.")
				break
			
			filepath_str: str = str(item_path)
			slug: str = item_path.stem  # Filename without extension as the resource slug

			self.logger.info(f"Attempting to upload: '{item_path.name}' as Nuclia resource with slug: '{slug}'")
			try:
				# `self.kb.upload_file` handles the creation of a new resource with the given slug
				# or updates an existing resource if the slug matches (behavior may depend on Nuclia KB settings).
				resource_creation_data = self.kb.upload_file(filepath=filepath_str, slug=slug)
				
				resource_id = getattr(resource_creation_data, 'uuid', None) or \
							  getattr(resource_creation_data, 'id', None)

				if resource_id:
					self.logger.info(f"Successfully uploaded '{item_path.name}'. Nuclia Resource Slug: '{slug}', RID: {resource_id}")
					upload_results[item_path.name] = {"status": "success", "resource_id": resource_id, "slug": slug}
				else:
					self.logger.warning(
						f"Upload of '{item_path.name}' (slug: '{slug}') accepted by Nuclia, but RID not in immediate response. "
						f"Response details: {str(resource_creation_data)}. Please verify in Nuclia dashboard."
					)
					upload_results[item_path.name] = {
						"status": "accepted_no_rid", 
						"slug": slug, 
						"details": f"Upload accepted, RID not in immediate response. Response: {str(resource_creation_data)}"
					}
			except Exception as e:
				self.logger.error(f"Unexpected error uploading '{item_path.name}' (slug: '{slug}'): {e}", exc_info=True)
				upload_results[item_path.name] = {"status": "failed", "slug": slug, "details": str(e)}
			
			files_processed_count += 1
		
		self.logger.info(f"Document upload process from folder '{folder_path}' finished for KB ID: '{self.kb.kbid}'.")
		return upload_results

	