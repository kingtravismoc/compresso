import subprocess, torch, torchvision.models as models, torchvision.transforms as transforms, numpy as np, os, time, PIL.Image, youtube_dl, ffmpeg, librosa, torchaudio, torchaudio.transforms as T, tempfile, warnings, collections, threading, queue, random, json, re, logging, argparse, glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# --- Entropy Generator ---
class EntropyPoolConfig:
    def __init__(self, min_pool_size, max_pool_size, generation_interval_sec):
        self.min_pool_size, self.max_pool_size, self.generation_interval_sec = min_pool_size, max_pool_size, generation_interval_sec

class EntropySource:
    def get_entropy(self, size: int) -> bytes: raise NotImplementedError("Subclasses must implement this method")

class FallbackEntropySource(EntropySource):
    def get_entropy(self, size: int) -> bytes:
        try: return os.urandom(size)
        except:
            logging.warning("os.urandom not available, falling back to less secure random module.")
            return bytes(random.getrandbits(8) for _ in range(size))

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
        while len(self._pool) < self.config.min_pool_size: self._generate_entropy()
        logging.info(f"Entropy pool initialized with {len(self._pool)} entries.")

    def _generate_entropy(self):
        with self._pool_lock:
            if len(self._pool) >= self.config.max_pool_size: return
        try:
            entropy_size = 32
            source = random.choice(self._sources)
            entropy = source.get_entropy(entropy_size)
            if entropy:
                with self._pool_lock: self._pool.append(entropy)
        except Exception as e: logging.error(f"Error generating entropy: {e}")

    def _timed_generation_task(self):
        while not self._stop_event.is_set():
            self._generate_entropy()
            self._stop_event.wait(self.config.generation_interval_sec)

    def start_generation(self):
        if hasattr(self, '_generation_thread') and self._generation_thread and self._generation_thread.is_alive(): return
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
            if self._generation_thread.is_alive(): logging.warning("Generation thread did not stop cleanly.")
            self._generation_thread = None
            logging.info("Entropy generation stopped.")
        else: logging.info("Entropy generation is not running.")

    def get_entropy(self, size: int) -> bytes | None:
        with self._pool_lock:
            if not self._pool: return None
            combined_entropy = b''
            while len(combined_entropy) < size and self._pool:
                 next_entry = self._pool.popleft()
                 combined_entropy += next_entry
            if len(combined_entropy) >= size: return combined_entropy[:size]
            else:
                if combined_entropy: self._pool.appendleft(combined_entropy)
                logging.warning(f"Requested {size} bytes but only {len(combined_entropy)} available in pool.")
                return None

    def get_pool_size(self) -> int:
        with self._pool_lock: return len(self._pool)

# --- Multigram LaTeX Engine Core ---
def multigrams(text, min_n=2, max_n=4, stride=1, char_level=True):
    out = []
    if char_level:
        seq = text.replace('\n',' ')
        for n in range(min_n, max_n+1):
            for i in range(0, len(seq)-n+1, stride): out.append(seq[i:i+n])
    else:
        toks = re.findall(r"\\w+|\\S", text)
        for n in range(min_n, max_n+1):
            for i in range(0, len(toks)-n+1, stride): out.append(' '.join(toks[i:i+n]))
    return out

class LatexTranscriber:
    def __init__(self): self.token_map, self.next_id = {}, 0
    def token_to_symbol(self, token):
        if token in self.token_map: return self.token_map[token]
        gid = self.next_id; name = "g" + self._base36(gid)
        self.next_id += 1
        latex = r"\mathcal{%s}" % name
        self.token_map[token] = latex
        return latex
    def _base36(self, n):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n == 0: return "0"
        s = ""; n_copy = n
        while n_copy: s = chars[n_copy % 36] + s; n_copy //= 36
        return s
    def transcribe_sequence(self, tokens):
        symbols = [self.token_to_symbol(t) for t in tokens]
        return " \\; ".join(symbols)

class Sequitur:
    def __init__(self): self.rules = {}; self.next_rule_id = 1
    def _new_rule_name(self): r = f"R{self.next_rule_id}"; self.next_rule_id += 1; return r
    def infer(self, sequence, max_rules=10000):
        seq = list(sequence); pair_counts = Counter(); positions = defaultdict(list)
        def rebuild_pairs():
            pair_counts.clear(); positions.clear()
            for i in range(len(seq)-1):
                p = (seq[i], seq[i+1]); pair_counts[p] += 1; positions[p].append(i)
        rebuild_pairs()
        rules = {}
        while True:
            if not pair_counts: break
            p, cnt = None, 0
            for k,v in pair_counts.items():
                if v > 1 and v > cnt: p, cnt = k, v
            if cnt <= 1 or len(rules) >= max_rules: break
            rule_name = self._new_rule_name()
            rules[rule_name] = [p[0], p[1]]
            new_seq = []; i = 0
            while i < len(seq):
                matched = False
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == p:
                    new_seq.append(rule_name); i += 2; matched = True
                else:
                    new_seq.append(seq[i]); i += 1
            seq = new_seq
            rebuild_pairs()
        grammar = {"S": seq}; grammar.update(rules); self.rules = grammar; return grammar

def cluster_rules_and_score(grammar, min_cluster_size=2, affinity='euclidean', threshold=0.5):
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
            cluster_scores[c] = 0.0; continue
        sub = vec_arr[idx]
        norm = np.linalg.norm(sub, axis=1, keepdims=True)
        subn = sub / (norm + 1e-12)
        sim = subn.dot(subn.T)
        n = sub.shape[0]
        if n > 1:
             avg_sim = (np.sum(sim) - n) / (n*(n-1))
             cluster_scores[c] = float(avg_sim)
        else: cluster_scores[c] = 0.0
    return cluster_map, cluster_scores

def grammar_to_latex(grammar, transcriber, cluster_map=None, cluster_scores=None, cluster_threshold=0.6):
    latex_defs = []
    keys = [k for k in grammar.keys() if k != "S"]
    valid_keys = [k for k in keys if re.match(r"R\d+", k)]; valid_keys.sort(key=lambda x: int(x[1:]))
    other_keys = [k for k in keys if not re.match(r"R\d+", k)]
    sorted_keys = valid_keys + other_keys

    cluster_to_macro = {}
    if cluster_map and cluster_scores:
        strong = [c for c,s in cluster_scores.items() if s >= cluster_threshold]
        for c in strong:
            macro = f"\\mathcal{{C{c}}}"; cluster_to_macro[c] = macro
            latex_defs.append(f"% Cluster {c} macro (score {cluster_scores.get(c, 0.0):.3f})")
            latex_defs.append(f"\\newcommand{{{macro}}}{{}} % placeholder for cluster-level macro")

    for r in sorted_keys:
        body = grammar.get(r, [])
        parts = []
        for tok in body:
            if tok in grammar: parts.append(f"\\{tok}")
            elif tok.startswith("<FOLD_"): parts.append(f"\\{tok[6:-1]}")
            else: parts.append(transcriber.token_to_symbol(tok))
        safe_rule_name = r.replace('_', '\\_')
        macro_def = f"\\newcommand{{\\{safe_rule_name}}}{{ {'\\; '.join(parts)} }}\n"
        latex_defs.append(macro_def)

    top = []
    s_body = grammar.get("S", [])
    for tok in s_body:
        if tok in grammar: top.append(f"\\{tok}")
        elif tok.startswith("<FOLD_"): top.append(f"\\{tok[6:-1]}")
        else: top.append(transcriber.token_to_symbol(tok))

    latex_main = "\\begin{document}\n" + " \\; ".join(top) + "\n\\end{document}\n"
    return "\n".join(latex_defs) + "\n\n" + latex_main

def iterative_engine(corpus_texts, iterations=4, min_token_freq=2):
    tokens = []
    for doc in corpus_texts:
        mg = multigrams(doc, min_n=2, max_n=4, stride=1, char_level=True)
        tokens.extend(mg)

    freq = Counter(tokens)
    tokens = [t for t in tokens if freq[t] >= min_token_freq]
    print(f"Initial token count (after filtering): {len(tokens)}")
    if not tokens:
        print("No valid tokens found after filtering. Stopping.")
        return []

    transcriber = LatexTranscriber()
    history = []
    seq = list(tokens)

    for it in range(iterations):
        print(f"=== Iteration {it+1} ===")
        if not seq: break

        seq_local = seq[:]
        sequitur = Sequitur()
        grammar = sequitur.infer(seq_local, max_rules=2000)
        print(f"Discovered rules: {len(grammar)-1}")

        if len(grammar) <= 1:
            print("No new rules discovered in this iteration. Stopping.")
            latex_doc = grammar_to_latex(grammar, transcriber)
            history.append({
                "iteration": it+1, "grammar": grammar, "latex": latex_doc,
                "seq_len_input": len(seq), "latex_len": len(latex_doc),
                "cluster_map": {}, "cluster_scores": {}
            })
            break

        cluster_map, cluster_scores = cluster_rules_and_score(grammar)
        print(f"Clustered rules: {len(set(cluster_map.values())) if cluster_map else 0}")

        latex_doc = grammar_to_latex(grammar, transcriber, cluster_map, cluster_scores)
        seq_len_input = len(seq)
        latex_len = len(latex_doc)
        print(f"Input Seq length (tokens/symbols): {seq_len_input}, Generated Latex bytes: {latex_len}")

        history.append({
            "iteration": it+1, "grammar": grammar, "latex": latex_doc,
            "seq_len_input": seq_len_input, "latex_len": latex_len,
            "cluster_map": cluster_map, "cluster_scores": cluster_scores
        })

        rule_freq = Counter()
        s_body = grammar.get("S", [])
        for sym in s_body: rule_freq[sym] += 1

        folded_symbols = set()
        if rule_freq:
            actual_rule_freq = {r: cnt for r, cnt in rule_freq.items() if r in grammar and r != "S"}
            median = np.median(list(actual_rule_freq.values())) if actual_rule_freq else 0

            for c in (cluster_scores.keys() if cluster_scores else []):
                if cluster_map.get(c) is not None and cluster_scores.get(c, 0.0) >= 0.55:
                     rules_in_cluster = [r for r, cid in cluster_map.items() if cid == c]
                     folded_symbols.update(rules_in_cluster)

            for sym, cnt in rule_freq.items():
                 is_rule = sym in grammar and sym != "S"
                 is_synthetic_fold = sym.startswith("<FOLD_")
                 if cnt > median or is_synthetic_fold:
                       if is_rule and len(grammar.get(sym, [])) > 1: folded_symbols.add(sym)
                       elif is_synthetic_fold: folded_symbols.add(sym)

            if not folded_symbols:
                print("No symbols identified for folding based on heuristics; stopping early.")
                break

            new_seq = []
            folding_map = {}
            fold_counter = 0

            for sym in seq:
                 if sym in folded_symbols:
                      if sym not in folding_map:
                           fold_counter += 1
                           folding_map[sym] = f"<FOLD_{fold_counter}>"
                      new_seq.append(folding_map[sym])
                 else:
                      new_seq.append(sym)

            seq = new_seq
            print(f"Symbols folded: {folded_symbols}")
            print(f"New sequence length for next iteration: {len(seq)}")
        else:
            print("No symbols found in top-level S body for frequency analysis; stopping.")
            break
    return history

def load_texts_from_path(path_pattern):
    files = glob.glob(path_pattern)
    docs = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf8', errors='ignore') as fh:
                docs.append(fh.read()[:100000])
            print(f"Loaded text from {f}")
        except Exception as e:
             print(f"Error loading file {f}: {e}")
    return docs

# --- Statistical Range Compression Simulation ---
def compress_with_statistical_range(data_string: str, entropy_generator: TrueEntropyGenerator, output_filename="compressed_archive.bin"):
    print("\n--- Starting Statistical Range Compression Simulation ---")
    entropy_key = entropy_generator.get_entropy(size=64)
    if not entropy_key:
        logging.error("Failed to acquire sufficient entropy for key generation. Aborting compression.")
        return False
    print(f"Acquired {len(entropy_key)} bytes of entropy for the compression key.")

    compression_seed = int.from_bytes(entropy_key[:4], byteorder='big')
    random.seed(compression_seed)
    simulated_compression_factor = 0.80
    print(f"Simulated statistical range based on entropy seed: {compression_seed}")
    print(f"Simulated compression factor applied: {simulated_compression_factor:.2f}")

    try:
        with open(output_filename, 'wb') as f:
            f.write(entropy_key)
            dummy_compressed_data = b'\x01\x02\x03' * (int(len(data_string) * simulated_compression_factor) // 3) + b'\x00'
            f.write(dummy_compressed_data)
        print(f"Simulated archive written to {output_filename}")
        print(f"Simulated Compressed Size: {len(dummy_compressed_data) + len(entropy_key)} bytes (approx. {int(len(data_string) * simulated_compression_factor)} bytes effective data).")
        print(f"Compression Ratio: {int(len(data_string) * simulated_compression_factor) / len(data_string.encode('utf-8')):.2f}")
        return True
    except Exception as e:
        logging.error(f"Error writing simulated archive: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    LATEX_ITERATIONS = 3 # Reduced for faster execution
    LATEX_MIN_TOKEN_FREQ = 2
    LATEX_INPUT_PATTERN = "sample_data/*"
    LATEX_OUTPUT_FILE = "generated_theory_minified.tex"
    ENTROPY_CONFIG = EntropyPoolConfig(min_pool_size=10, max_pool_size=50, generation_interval_sec=1.0)
    COMPRESSION_ARCHIVE = "final_compressed_archive.bin"
    
    # 1. Entropy Generation
    entropy_generator = TrueEntropyGenerator(ENTROPY_CONFIG)
    entropy_generator.start_generation()
    time.sleep(1.5)

    # 2. Simulate Data Input (Since video processing is stubbed/complex)
    corpus_texts = [
        """
        This is a small sample document. It contains several words. 这是一个示例文本。
        It is a sample document with some repeated patterns. sample document.
        Another repeated phrase: pattern A B C. pattern A B C.
        """
    ]
    csv_file_path = "/content/sample_data/california_housing_train.csv"
    if os.path.exists(csv_file_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            corpus_texts = [df.to_string()]
            logging.info(f"Loaded text from {csv_file_path}")
        except Exception as e:
            logging.warning(f"Error loading or processing CSV {csv_file_path}: {e}. Using fallback text.")
    else:
        logging.warning(f"No sample data found at {csv_file_path}. Using internal fallback text.")

    # 3. Run Grammar Induction
    print("\n--- Running Grammar Induction ---")
    history = iterative_engine(corpus_texts, iterations=LATEX_ITERATIONS, min_token_freq=LATEX_MIN_TOKEN_FREQ)

    # 4. Export LaTeX
    if history:
        latest = history[-1]
        try:
            with open(LATEX_OUTPUT_FILE, 'w', encoding='utf8') as fh:
                fh.write(latest["latex"])
            print(f"Wrote final LaTeX to {LATEX_OUTPUT_FILE}")
        except Exception as e:
            logging.error(f"Error writing LaTeX file {LATEX_OUTPUT_FILE}: {e}")
        latex_content = latest["latex"]
        original_size_bytes = len(latex_content.encode('utf-8'))
    else:
        print("No history generated. Cannot proceed with compression.")
        latex_content = None
        original_size_bytes = 0

    # 5. Simulate Compression & Integrate Entropy
    if latex_content:
        compress_with_statistical_range(latex_content, entropy_generator, COMPRESSION_ARCHIVE)

    # 6. Finalize
    entropy_generator.stop_generation()
    print("\n--- Minified Script Execution Complete ---")
