import subprocess
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import time
from PIL import Image
import youtube_dl
import ffmpeg
import librosa
import torchaudio
import torchaudio.transforms as T
import tempfile
import warnings
import collections
import threading
import queue
import random
import json
import re
import logging
import argparse
import glob # For the LaTeX engine file loading

# --- Configuration and Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# GLOBAL/SHARED VARIABLES (Initialized later)
# entropy_generator = None
# history_manager = None
# completed_videos_queue = None
# association_results_queue = None
# text_vectorizer = None
# time_window = 2.0
# dummy_video_path = "/content/dummy_video.mp4"

# --- Video and Audio Ingestion (Simplified/Removed for Direct Text Input) ---
# Keeping the functions for completeness but they won't be the main focus of the final run.

def download_youtube_video(url, output_path):
    """Downloads a YouTube video. (Function kept for context)"""
    logging.warning("YouTube download function stubbed: Not executed in final pipeline.")
    return None

def process_webm_file(input_path, output_path):
    """Processes a webm file. (Function kept for context)"""
    logging.warning("WebM processing function stubbed: Not executed in final pipeline.")
    return None

# --- Frame Extraction and Vectorization ---

def extract_and_vectorize_frames(video_path, frame_interval_n=10):
    """
    Extracts frames from a video at a specified interval and vectorizes them
    using a pre-trained vision model. (Function kept for context)
    """
    logging.warning("Frame extraction function stubbed: Not executed in final pipeline due to primary focus on LaTeX engine.")
    # Placeholder to allow Orchestrator to run without actual video files
    return [{'vector': np.random.rand(512), 'metadata': {'source': 'video_frame', 'filename': f"frame_{i}.jpg", 'timestamp': i * frame_interval_n}} for i in range(5)]


# --- Audio Extraction and Vectorization ---

def extract_and_vectorize_audio(video_path, segment_duration_sec=10):
    """
    Extracts audio from a video, segments it, and vectorizes each segment. (Function kept for context)
    """
    logging.warning("Audio extraction/vectorization function stubbed: Not executed in final pipeline due to primary focus on LaTeX engine.")
    # Placeholder to allow Orchestrator to run without actual video files
    return [{'vector': np.random.rand(512), 'metadata': {'source': 'audio_segment', 'start_time_sec': i * segment_duration_sec}} for i in range(3)]


# --- Subtitle Extraction and Vectorization ---

# Placeholder for TextVectorizer initialization (needed by subtitle function)
try:
    from transformers import AutoTokenizer, AutoModel
    class TextVectorizer:
        def __init__(self, model_name='bert-base-uncased'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"TextVectorizer initialized with {model_name} on {self.device}.")

        def vectorize_text(self, text, metadata={}):
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return vector
    text_vectorizer = TextVectorizer()
except ImportError:
    logging.warning("Transformers library not found. Using dummy TextVectorizer.")
    class TextVectorizer:
        def __init__(self, model_name=None, dimensions=768):
            self.vector_dimensions = dimensions
            logging.warning(f"Dummy TextVectorizer initialized. Dimensions: {self.vector_dimensions}")
        def vectorize_text(self, text, metadata={}):
            return np.random.rand(self.vector_dimensions)
    text_vectorizer = TextVectorizer()


def extract_and_vectorize_subtitles(video_path, text_vectorizer_instance):
    """
    Extracts and vectorizes subtitles from a video file. (Function kept for context)
    """
    logging.warning("Subtitle extraction function stubbed: Not executed in final pipeline due to primary focus on LaTeX engine.")
    # Placeholder to allow Orchestrator to run without actual video files
    return [{'vector': np.random.rand(512), 'metadata': {'source': 'subtitle', 'start_time_sec': i * 1.5, 'end_time_sec': i*1.5 + 1.0, 'text': 'dummy subtitle text'}} for i in range(2)]


# --- AI Vision Scoring ---
# (Stubs for scoring model loading and function)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    scoring_model = models.resnet18(pretrained=True).eval().to(device)
    score_preprocess = transforms.Compose([...]) # Full definition omitted for brevity
    imagenet_classes = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead']
except Exception:
    logging.warning("Scoring model loading skipped/failed for brevity/no video context.")
    scoring_model = None
    score_preprocess = None
    imagenet_classes = []

def score_frame_vector(frame_data):
    """
    Scores a frame based on its associated data. (Function kept for context)
    """
    if scoring_model is None or score_preprocess is None:
        return None
    # Simplified for no-video context: assign a random score based on the frame's timestamp/index
    if 'timestamp' in frame_data['metadata']:
        seed = int(frame_data['metadata']['timestamp'] * 100)
        random.seed(seed)
        prob = random.random()
        return {'source': 'vision_score', 'model': 'ResNet18', 'analysis_type': 'simulated', 'probability': prob}
    return {'source': 'vision_score', 'model': 'ResNet18', 'analysis_type': 'simulated', 'probability': 0.5}


# --- Vector Interpolation ---

def interpolate_frame_vectors(frame_data_list, frame_interval_n):
    """
    Interpolates vectors between extracted frames based on their timestamps. (Function kept for context)
    """
    if not frame_data_list or len(frame_data_list) < 2:
        return frame_data_list

    frame_data_list.sort(key=lambda x: x['metadata'].get('timestamp', 0))
    interpolated_vectors = [frame_data_list[0]]

    for i in range(len(frame_data_list) - 1):
        frame1 = frame_data_list[i]
        frame2 = frame_data_list[i+1]
        vec1 = frame1['vector']
        vec2 = frame2['vector']
        timestamp1 = frame1['metadata'].get('timestamp')
        timestamp2 = frame2['metadata'].get('timestamp')

        if timestamp1 is None or timestamp2 is None or timestamp2 <= timestamp1:
            interpolated_vectors.append(frame2)
            continue

        time_diff = timestamp2 - timestamp1
        num_interpolation_points = int(time_diff) - 1

        if num_interpolation_points > 0:
            for j in range(1, num_interpolation_points + 1):
                interpolation_factor = j / time_diff
                interpolated_vector = vec1 + (vec2 - vec1) * interpolation_factor

                interpolated_vectors.append({
                    'vector': interpolated_vector,
                    'metadata': {
                        'source': 'interpolated_frame',
                        'timestamp': timestamp1 + j,
                        'interpolated_from_timestamp_1': timestamp1,
                        'interpolated_from_timestamp_2': timestamp2,
                        'interpolation_factor': interpolation_factor
                    }
                })
        interpolated_vectors.append(frame2)

    interpolated_vectors.sort(key=lambda x: x['metadata'].get('timestamp', 0))
    return interpolated_vectors


# --- Multimodal Vector Association ---

def associate_multimodal_vectors(frame_vectors, audio_vectors, subtitle_vectors, time_window_sec=2.0):
    """
    Associates vectors from different modalities based on a time window.
    """
    all_vectors = []
    for vec_data in frame_vectors:
        timestamp = vec_data['metadata'].get('timestamp')
        if timestamp is not None:
            all_vectors.append({'type': 'frame', 'timestamp': timestamp, 'data': vec_data})

    for vec_data in audio_vectors:
        timestamp = vec_data['metadata'].get('start_time_sec')
        if timestamp is not None:
            all_vectors.append({'type': 'audio', 'timestamp': timestamp, 'data': vec_data})

    for vec_data in subtitle_vectors:
        timestamp = vec_data['metadata'].get('start_time_sec')
        if timestamp is not None:
            all_vectors.append({'type': 'subtitle', 'timestamp': timestamp, 'data': vec_data})

    all_vectors.sort(key=lambda x: x['timestamp'])

    multimodal_associations = defaultdict(lambda: {'frames': [], 'audio': [], 'subtitles': []})

    current_window_start_time = 0
    window_end_time = time_window_sec

    for vec_entry in all_vectors:
        timestamp = vec_entry['timestamp']

        while timestamp >= window_end_time:
            current_window_start_time = window_end_time
            window_end_time += time_window_sec

        window_key = current_window_start_time

        if vec_entry['type'] == 'frame':
            multimodal_associations[window_key]['frames'].append(vec_entry['data'])
        elif vec_entry['type'] == 'audio':
            multimodal_associations[window_key]['audio'].append(vec_entry['data'])
        elif vec_entry['type'] == 'subtitle':
            multimodal_associations[window_key]['subtitles'].append(vec_entry['data'])

    return multimodal_associations


# --- Entropy Generation (TrueEntropyGenerator) ---

class EntropyPoolConfig:
    def __init__(self, min_pool_size, max_pool_size, generation_interval_sec):
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.generation_interval_sec = generation_interval_sec

class EntropySource:
    def get_entropy(self, size: int) -> bytes:
        raise NotImplementedError("Subclasses must implement this method")

class FallbackEntropySource(EntropySource):
    def get_entropy(self, size: int) -> bytes:
        try:
            return os.urandom(size)
        except NotImplementedError:
            logging.warning("os.urandom not available, falling back to less secure random module.")
            return bytes(random.getrandbits(8) for _ in range(size))
        except Exception as e:
            logging.error(f"Error in FallbackEntropySource: {e}")
            return b''

class TrueEntropyGenerator:
    def __init__(self, config: EntropyPoolConfig, sources: list[EntropySource] = None):
        self.config = config
        self._pool: collections.deque[bytes] = collections.deque()
        self._sources = sources if sources is not None and len(sources) > 0 else [FallbackEntropySource()]
        self._generation_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pool_lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        logging.info("Initializing entropy pool...")
        while len(self._pool) < self.config.min_pool_size:
            self._generate_entropy()
        logging.info(f"Entropy pool initialized with {len(self._pool)} entries.")

    def _generate_entropy(self):
        with self._pool_lock:
            if len(self._pool) >= self.config.max_pool_size:
                return
        try:
            entropy_size = 32
            source = random.choice(self._sources)
            entropy = source.get_entropy(entropy_size)
            if entropy:
                with self._pool_lock:
                    self._pool.append(entropy)
        except Exception as e:
            logging.error(f"Error generating entropy: {e}")

    def _timed_generation_task(self):
        while not self._stop_event.is_set():
            self._generate_entropy()
            self._stop_event.wait(self.config.generation_interval_sec)

    def start_generation(self):
        if hasattr(self, '_generation_thread') and self._generation_thread and self._generation_thread.is_alive():
             logging.info("Entropy generation already running.")
             return
        logging.info(f"Entropy generation starting with interval {self.config.generation_interval_sec} seconds.")
        self._stop_event.clear()
        self._generation_thread = threading.Thread(target=self._timed_generation_task)
        self._generation_thread.daemon = True
        self._generation_thread.start()

    def stop_generation(self):
        if hasattr(self, '_generation_thread') and self._generation_thread and self._generation_thread.is_alive():
            logging.info("Signalling entropy generation thread to stop.")
            self._stop_event.set()
            self._generation_thread.join(timeout=2.0)
            if self._generation_thread.is_alive():
                logging.warning("Generation thread did not stop cleanly.")
            self._generation_thread = None
            logging.info("Entropy generation stopped.")
        else:
            logging.info("Entropy generation is not running.")

    def get_entropy(self, size: int) -> bytes | None:
        with self._pool_lock:
            if not self._pool:
                return None
            combined_entropy = b''
            while len(combined_entropy) < size and self._pool:
                 next_entry = self._pool.popleft()
                 combined_entropy += next_entry

            if len(combined_entropy) >= size:
                return combined_entropy[:size]
            else:
                if combined_entropy:
                     self._pool.appendleft(combined_entropy)
                logging.warning(f"Requested {size} bytes but only {len(combined_entropy)} available in pool.")
                return None

    def get_pool_size(self) -> int:
        with self._pool_lock:
            return len(self._pool)


# --- Threaded Processing Orchestration ---

class VideoProcessingOrchestrator:
    def __init__(self, video_path, text_vectorizer_instance, completed_videos_queue, frame_interval=10, audio_segment_duration=10):
        if not os.path.exists(video_path):
             logging.error(f"Error: Video file not found during Orchestrator init: {video_path}")
             self._initialization_failed = True
             self.video_path = video_path
             self.text_vectorizer_instance = None
             self.completed_videos_queue = None
             self.frame_interval = frame_interval
             self.audio_segment_duration = audio_segment_duration
             self._frame_results_queue = None
             self._audio_results_queue = None
             self._subtitle_results_queue = None
             self._threads = []
             return

        self._initialization_failed = False
        self.video_path = video_path
        self.text_vectorizer_instance = text_vectorizer_instance
        self.completed_videos_queue = completed_videos_queue
        self.frame_interval = frame_interval
        self.audio_segment_duration = audio_segment_duration

        self._frame_results_queue = queue.Queue()
        self._audio_results_queue = queue.Queue()
        self._subtitle_results_queue = queue.Queue()
        self._threads = []
        logging.info(f"Initialized orchestrator for video: {self.video_path}")

    def _run_frame_processing(self):
        try:
            frames = extract_and_vectorize_frames(self.video_path, self.frame_interval)
            interpolated_frames = interpolate_frame_vectors(frames, self.frame_interval)
            scored_frames = []
            for frame_data in interpolated_frames:
                 score_results = score_frame_vector(frame_data)
                 if score_results:
                      frame_data['vision_score'] = score_results
                 scored_frames.append(frame_data)
            self._frame_results_queue.put(scored_frames)
        except Exception as e:
            logging.error(f"Error in frame processing for {self.video_path}: {e}")
            self._frame_results_queue.put([])

    def _run_audio_processing(self):
        try:
            audio_data = extract_and_vectorize_audio(self.video_path, self.audio_segment_duration)
            self._audio_results_queue.put(audio_data)
        except Exception as e:
            logging.error(f"Error in audio processing for {self.video_path}: {e}")
            self._audio_results_queue.put([])

    def _run_subtitle_processing(self):
        try:
            subtitles = extract_and_vectorize_subtitles(self.video_path, self.text_vectorizer_instance)
            self._subtitle_results_queue.put(subtitles)
        except Exception as e:
            logging.error(f"Error in subtitle processing for {self.video_path}: {e}")
            self._subtitle_results_queue.put([])

    def start_processing(self):
        if self._initialization_failed:
             logging.info(f"Skipping processing for {self.video_path} due to initialization failure.")
             if self.completed_videos_queue:
                 self.completed_videos_queue.put({
                     'video_path': self.video_path,
                     'frames': [], 'audio': [], 'subtitles': [],
                     'error': 'Initialization failed: Video file not found.'
                 })
             return

        logging.info(f"Starting concurrent processing threads for {self.video_path}...")
        frame_thread = threading.Thread(target=self._run_frame_processing)
        audio_thread = threading.Thread(target=self._run_audio_processing)
        subtitle_thread = threading.Thread(target=self._run_subtitle_processing)
        self._threads = [frame_thread, audio_thread, subtitle_thread]
        for thread in self._threads:
            thread.start()

    def join_threads(self):
        if self._initialization_failed:
             return
        for thread in self._threads:
            thread.join()

    def collect_results(self):
        if self._initialization_failed:
            return None
        logging.info(f"Collecting results for {self.video_path}...")
        processed_data = {
            'video_path': self.video_path,
            'frames': [], 'audio': [], 'subtitles': [], 'error': None
        }
        try:
            processed_data['frames'] = self._frame_results_queue.get()
            processed_data['audio'] = self._audio_results_queue.get()
            processed_data['subtitles'] = self._subtitle_results_queue.get()
            logging.info(f"Collected all modality results for {self.video_path}.")
        except queue.Empty:
            logging.error(f"Error collecting results for {self.video_path}: One or more internal queues are empty.")
            processed_data['error'] = 'Result collection failed: Internal queue empty.'
        except Exception as e:
             logging.error(f"An unexpected error occurred while collecting results for {self.video_path}: {e}")
             processed_data['error'] = f'Unexpected error during collection: {e}'

        if self.completed_videos_queue:
            self.completed_videos_queue.put(processed_data)
            logging.info(f"Put completed processing data for {self.video_path} into association queue.")
        else:
             logging.error(f"Completed videos queue not initialized for {self.video_path}. Cannot put results.")

        return processed_data


# --- Dedicated Thread for Association ---

class AssociationThread(threading.Thread):
    def __init__(self, completed_videos_queue, association_results_queue, time_window_sec=2.0):
        super().__init__()
        self.completed_videos_queue = completed_videos_queue
        self.association_results_queue = association_results_queue
        self.time_window_sec = time_window_sec
        self._stop_event = threading.Event()
        logging.info("AssociationThread initialized.")

    def run(self):
        logging.info("AssociationThread started.")
        while not self._stop_event.is_set() or not self.completed_videos_queue.empty():
            try:
                video_data = self.completed_videos_queue.get(timeout=1.0)
                video_path = video_data.get('video_path', 'Unknown')

                if video_data.get('error'):
                     logging.warning(f"AssociationThread: Skipping association for {video_path} due to processing error: {video_data['error']}")
                     self.association_results_queue.put({
                         'video_path': video_path,
                         'associations': {},
                         'error': video_data['error']
                     })
                     self.completed_videos_queue.task_done()
                     continue

                frames = video_data.get('frames', [])
                audio = video_data.get('audio', [])
                subtitles = video_data.get('subtitles', [])

                if not frames and not audio and not subtitles:
                      logging.warning(f"AssociationThread: No data found for video {video_path}. Skipping association.")
                      self.association_results_queue.put({
                          'video_path': video_path,
                          'associations': {},
                          'error': 'No multimodal data processed'
                      })
                      self.completed_videos_queue.task_done()
                      continue

                try:
                    multimodal_associations = associate_multimodal_vectors(
                        frames, audio, subtitles, time_window_sec=self.time_window_sec
                    )
                    self.association_results_queue.put({
                        'video_path': video_path,
                        'associations': multimodal_associations
                    })
                except Exception as e:
                    logging.error(f"AssociationThread: Error during association for {video_path}: {e}")
                    self.association_results_queue.put({
                         'video_path': video_path,
                         'associations': {},
                         'error': f'Association failed: {str(e)}'
                     })
                self.completed_videos_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"AssociationThread: An unexpected error occurred in main loop: {e}")

        logging.info("AssociationThread stopping.")

    def stop(self):
        self._stop_event.set()


# --- Learning and Improvement ---

class LearningHistory:
    def __init__(self, history_file=None):
        self.associations_history = []
        self.vision_scores_history = []
        self.feedback_data = []
        self.history_file = history_file
        self._load_history()

    def add_association_result(self, association_result):
        if any(res.get('video_path') == association_result.get('video_path') for res in self.associations_history):
            return
        self.associations_history.append(association_result)
        logging.info(f"Added association result for video: {association_result.get('video_path', 'Unknown')}")
        self._save_history()

    def add_vision_scores(self, video_path, frame_vision_scores):
        if any(res.get('video_path') == video_path for res in self.vision_scores_history):
             return
        self.vision_scores_history.append({'video_path': video_path, 'scores': frame_vision_scores})
        logging.info(f"Added {len(frame_vision_scores)} vision scores for video: {video_path}")
        self._save_history()

    def simulate_feedback(self, video_path, feedback_type, feedback_details):
        feedback_entry = {'timestamp': time.time(), 'video_path': video_path, 'type': feedback_type, 'details': feedback_details}
        self.feedback_data.append(feedback_entry)
        logging.info(f"Simulated feedback received for {video_path}: {feedback_type}")
        self._save_history()

    def perform_learning_analysis(self):
        logging.info("\nPerforming learning analysis on history...")
        total_associations = sum(len(res.get('associations', {})) for res in self.associations_history)
        total_frames_scored = sum(len(res.get('scores', [])) for res in self.vision_scores_history)
        total_feedback_entries = len(self.feedback_data)
        logging.info(f"  - Total association windows stored: {total_associations}")
        logging.info(f"  - Total frames with vision scores stored: {total_frames_scored}")
        logging.info(f"  - Total feedback entries stored: {total_feedback_entries}")

        window_interest_scores = defaultdict(int)
        for fb in self.feedback_data:
             if fb['type'] == 'interesting_segment' and 'window_key' in fb['details'] and 'video_path' in fb:
                  window_key = fb['details']['window_key']
                  video_path = fb['video_path']
                  window_interest_scores[f"{video_path}_{window_key}"] += 1

        if window_interest_scores:
            logging.info(f"  - Interest scores for windows based on feedback: {dict(window_interest_scores)}")

        logging.info("Learning analysis complete (basic).")

    def _save_history(self):
        if self.history_file:
            try:
                data_to_save = {'associations': self.associations_history, 'vision_scores': self.vision_scores_history, 'feedback': self.feedback_data}
                def prepare_for_json(data):
                    if isinstance(data, dict): return {k: prepare_for_json(v) for k, v in data.items()}
                    elif isinstance(data, list): return [prepare_for_json(item) for item in data]
                    elif isinstance(data, np.ndarray): return data.tolist()
                    else: return data
                serializable_data = prepare_for_json(data_to_save)
                with open(self.history_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
            except Exception as e:
                logging.error(f"Error saving history to {self.history_file}: {e}")

    def _load_history(self):
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    loaded_data = json.load(f)
                self.associations_history = loaded_data.get('associations', [])
                self.vision_scores_history = loaded_data.get('vision_scores', [])
                self.feedback_data = loaded_data.get('feedback', [])
                logging.info(f"History loaded from {self.history_file}.")
            except Exception as e:
                logging.error(f"Error loading history from {self.history_file}: {e}")
                self.associations_history, self.vision_scores_history, self.feedback_data = [], [], []


# --- Result Output ---

def export_results(video_path, processed_data, associations, output_dir="/content/processed_results"):
    """Exports the processed multimodal data and associations to a JSON file."""
    logging.info("\nExporting processed results...")
    os.makedirs(output_dir, exist_ok=True)
    export_data = {
        'video_path': video_path,
        'processed_vectors': {'frames': processed_data.get('frames', []),
                              'audio': processed_data.get('audio', []),
                              'subtitles': processed_data.get('subtitles', [])},
        'multimodal_associations': associations,
    }

    def convert_numpy_to_list_recursive(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert_numpy_to_list_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_numpy_to_list_recursive(item) for item in obj]
        else: return obj

    serializable_data = convert_numpy_to_list_recursive(export_data)
    timestamp = int(time.time())
    filename = f"processed_{os.path.basename(video_path).replace('.', '_')}_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)

    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        logging.info(f"Processed results successfully exported to {file_path}")
    except Exception as e:
        logging.error(f"Error during export for {video_path}: {e}")


# --- Multigram LaTeX Engine Logic (Adapted from cell_id: 77c0d43f) ---

class Sequitur:
    # ... (All Sequitur class methods as defined in cell_id: 77c0d43f)
    def __init__(self):
        self.rules = {}
        self.next_rule_id = 1

    def _new_rule_name(self):
        r = f"R{self.next_rule_id}"
        self.next_rule_id += 1
        return r

    def infer(self, sequence, max_rules=10000):
        seq = list(sequence)
        pair_counts = Counter()
        positions = defaultdict(list)

        def rebuild_pairs():
            pair_counts.clear()
            positions.clear()
            for i in range(len(seq)-1):
                p = (seq[i], seq[i+1])
                pair_counts[p] += 1
                positions[p].append(i)

        rebuild_pairs()
        rules = {}
        while True:
            if not pair_counts: break
            p, cnt = None, 0
            for k,v in pair_counts.items():
                if v > 1 and v > cnt:
                    p, cnt = k, v
            if cnt <= 1 or len(rules) >= max_rules: break
            rule_name = self._new_rule_name()
            rules[rule_name] = [p[0], p[1]]
            new_seq = []
            i = 0
            while i < len(seq):
                matched = False
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == p:
                    new_seq.append(rule_name)
                    i += 2
                    matched = True
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
            rebuild_pairs()
        grammar = {"S": seq}
        grammar.update(rules)
        self.rules = grammar
        return grammar

class LatexTranscriber:
    # ... (All LatexTranscriber methods as defined in cell_id: 77c0d43f)
    def __init__(self):
        self.token_map = {}
        self.next_id = 0

    def token_to_symbol(self, token):
        if token in self.token_map:
            return self.token_map[token]
        gid = self.next_id
        name = "g" + self._base36(gid)
        self.next_id += 1
        latex = r"\mathcal{%s}" % name
        self.token_map[token] = latex
        return latex

    def _base36(self, n):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n == 0: return "0"
        s = ""
        while n:
            s = chars[n % 36] + s
            n //= 36
        return s

    def transcribe_sequence(self, tokens):
        symbols = [self.token_to_symbol(t) for t in tokens]
        return " \\; ".join(symbols)

def cluster_rules_and_score(grammar, min_cluster_size=2, affinity='euclidean', threshold=0.5):
    # ... (All cluster_rules_and_score methods as defined in cell_id: 77c0d43f)
    rules = {k:v for k,v in grammar.items() if k != "S"}
    labels = list(rules.keys())
    texts = [" ".join(v) for v in rules.values()]
    if len(texts) < 2: return {}, {}
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2,4)).fit_transform(texts)
    vec_arr = vec.toarray()
    n_clusters = max(2, int(math.sqrt(len(texts))))
    n_clusters = min(n_clusters, vec_arr.shape[0])
    if n_clusters < 2: return {}, {}
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(vec_arr)
    cluster_map = {label: clustering.labels_[i] for i,label in enumerate(labels)}
    cluster_scores = {}
    for c in set(clustering.labels_):
        idx = [i for i,l in enumerate(clustering.labels_) if l==c]
        if len(idx) < min_cluster_size:
            cluster_scores[c] = 0.0
            continue
        sub = vec_arr[idx]
        norm = np.linalg.norm(sub, axis=1, keepdims=True)
        subn = sub / (norm + 1e-12)
        sim = subn.dot(subn.T)
        n = sub.shape[0]
        if n > 1:
             avg_sim = (np.sum(sim) - n) / (n*(n-1))
             cluster_scores[c] = float(avg_sim)
        else:
             cluster_scores[c] = 0.0
    return cluster_map, cluster_scores

def grammar_to_latex(grammar, transcriber, cluster_map=None, cluster_scores=None, cluster_threshold=0.6):
    # ... (All grammar_to_latex methods as defined in cell_id: 77c0d43f)
    latex_defs = []
    keys = [k for k in grammar.keys() if k != "S"]
    valid_keys = [k for k in keys if re.match(r"R\d+", k)]
    valid_keys.sort(key=lambda x: int(x[1:]))
    other_keys = [k for k in keys if not re.match(r"R\d+", k)]
    sorted_keys = valid_keys + other_keys

    cluster_to_macro = {}
    if cluster_map and cluster_scores:
        strong = [c for c,s in cluster_scores.items() if s >= cluster_threshold]
        for c in strong:
            macro = f"\\mathcal{{C{c}}}"
            cluster_to_macro[c] = macro
            latex_defs.append(f"% Cluster {c} macro (score {cluster_scores.get(c, 0.0):.3f})")
            latex_defs.append(f"\\newcommand{{{macro}}}{{}} % placeholder for cluster-level macro")

    for r in sorted_keys:
        body = grammar.get(r, [])
        parts = []
        for tok in body:
            if tok in grammar:
                parts.append(f"\\{tok}")
            elif tok.startswith("<FOLD_"):
                 original_rule_name = tok[6:-
