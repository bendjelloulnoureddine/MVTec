import sqlite3
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict
import json

class InferenceDatabase:
    def __init__(self, db_path: str = "inference_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create inference_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inference_id TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                input_image_path TEXT NOT NULL,
                output_folder TEXT NOT NULL,
                model_path TEXT NOT NULL,
                anomaly_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_inference_result(self, 
                            model_type: str,
                            input_image_path: str,
                            output_folder: str,
                            model_path: str,
                            anomaly_score: Optional[float] = None,
                            metadata: Optional[Dict] = None) -> str:
        """Save inference result to database"""
        inference_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO inference_results 
            (inference_id, model_type, input_image_path, output_folder, model_path, anomaly_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (inference_id, model_type, input_image_path, output_folder, model_path, anomaly_score, metadata_json))
        
        conn.commit()
        conn.close()
        
        return inference_id
    
    def save_model_info(self, 
                       model_name: str,
                       model_type: str,
                       model_path: str,
                       performance_metrics: Optional[Dict] = None) -> int:
        """Save model information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_json = json.dumps(performance_metrics) if performance_metrics else None
        
        cursor.execute('''
            INSERT INTO models (model_name, model_type, model_path, performance_metrics)
            VALUES (?, ?, ?, ?)
        ''', (model_name, model_type, model_path, metrics_json))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return model_id
    
    def get_inference_results(self, limit: int = 50) -> List[Dict]:
        """Get recent inference results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM inference_results 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_model_info(self, model_id: int) -> Optional[Dict]:
        """Get model information by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        
        conn.close()
        return None
    
    def get_inference_by_id(self, inference_id: str) -> Optional[Dict]:
        """Get inference result by inference ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM inference_results WHERE inference_id = ?', (inference_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        
        conn.close()
        return None