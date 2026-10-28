"""
TextGuard AI - Analysis History Manager
Provides persistent local storage for analysis sessions using JSON file handling.
Respects user privacy settings and allows searching, deleting, and clearing history.
"""

import datetime
import json
import os
import uuid
from typing import Dict, Any, List, Optional


DEFAULT_HISTORY_FILE = os.path.join("data", "history.json")


class HistoryManager:
    """
    Manages persistent analysis records on disk.
    Satisfies requirements for persistent storage (FR-3) and user data privacy controls.
    """

    def __init__(self, filepath: str = DEFAULT_HISTORY_FILE):
        self.filepath = filepath
        self.settings = {
            "save_history_enabled": True,
            "store_full_text": True,  # If False, stores only a 100-character preview
        }
        self.records: List[Dict[str, Any]] = []
        self._ensure_storage_ready()

    def _ensure_storage_ready(self) -> None:
        """
        Verify that directory exists and load existing history file.
        If file does not exist, initialize with structured storage.
        """
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            self._save_to_file()
        else:
            self.load()

    def load(self) -> bool:
        """
        Load records and user settings from JSON file.
        """
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
                self.records = payload.get("records", [])
                self.settings.update(payload.get("settings", {}))
            return True
        except Exception as e:
            print(f"[!] Warning: Failed to load history from {self.filepath}: {e}")
            self.records = []
            return False

    def _save_to_file(self) -> bool:
        """
        Write current records and settings to JSON file.
        """
        try:
            payload = {
                "version": "1.0.0",
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "settings": self.settings,
                "records": self.records,
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[!] Error: Could not save history to {self.filepath}: {e}")
            return False

    def add_record(
        self,
        raw_text: str,
        prediction_result: Dict[str, Any],
        linguistic_stats: Dict[str, Any],
    ) -> Optional[str]:
        """
        Create and append a new analysis record to persistent history.

        Args:
            raw_text (str): Input text analyzed.
            prediction_result (dict): ML classification output.
            linguistic_stats (dict): Extracted stylometric metrics.

        Returns:
            Optional[str]: Unique record ID if saved, None if history disabled.
        """
        if not self.settings.get("save_history_enabled", True):
            return None

        record_id = f"TG-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        preview = raw_text[:120].replace("\n", " ").strip()
        if len(raw_text) > 120:
            preview += "..."

        stored_text = raw_text if self.settings.get("store_full_text", True) else preview

        record = {
            "id": record_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text_preview": preview,
            "text_content": stored_text,
            "word_count": linguistic_stats.get("word_count", 0),
            "char_count": linguistic_stats.get("char_count_total", 0),
            "sentence_count": linguistic_stats.get("sentence_count", 0),
            "verdict": prediction_result.get("verdict", "Unknown"),
            "verdict_code": prediction_result.get("verdict_code", "UNKNOWN"),
            "ai_probability": prediction_result.get("ai_probability", 0.0),
            "human_probability": prediction_result.get("human_probability", 0.0),
            "confidence_level": prediction_result.get("confidence_level", "Unknown"),
            "lexical_diversity": linguistic_stats.get("lexical_diversity_ttr", 0.0),
            "burstiness_std": linguistic_stats.get("sentence_length_std", 0.0),
            "reading_ease": linguistic_stats.get("flesch_reading_ease", 0.0),
        }

        # Prepend so newest is first
        self.records.insert(0, record)
        self._save_to_file()
        return record_id

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return list of all records."""
        return list(self.records)

    def search_records(self, query: str) -> List[Dict[str, Any]]:
        """
        Search records matching query in text preview, verdict, or ID.
        """
        if not query:
            return self.get_all_records()
        q = query.lower().strip()
        return [
            r for r in self.records
            if q in r.get("id", "").lower()
            or q in r.get("text_preview", "").lower()
            or q in r.get("verdict", "").lower()
            or q in r.get("timestamp", "").lower()
        ]

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single record by its unique ID."""
        for r in self.records:
            if r.get("id") == record_id:
                return r
        return None

    def delete_record(self, record_id: str) -> bool:
        """Delete a single record by ID."""
        initial_count = len(self.records)
        self.records = [r for r in self.records if r.get("id") != record_id]
        if len(self.records) < initial_count:
            self._save_to_file()
            return True
        return False

    def clear_all_history(self) -> bool:
        """Delete all stored records."""
        self.records = []
        return self._save_to_file()

    def update_privacy_settings(self, save_enabled: bool, full_text_enabled: bool) -> None:
        """Update user privacy preferences."""
        self.settings["save_history_enabled"] = bool(save_enabled)
        self.settings["store_full_text"] = bool(full_text_enabled)
        self._save_to_file()
