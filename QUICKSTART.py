
"""
Quick Start Guide for Enhanced IndicTrans2
Run this in Jupyter/Colab to test locally before deploying to Streamlit
"""

# ============================================================================
# SETUP
# ============================================================================
!pip install -q torch transformers datasets streamlit
!pip install -q indic-nlp-library IndicTransToolkit sacrebleu evaluate

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import sys
sys.path.insert(0, '/content')  # Or your local path

# Import our novelty modules
from enhanced_streamlit_app import (
    StructurePreserver, NamedEntityTracker, ContextAwareCache,
    QualityMetrics, ErrorDetector, DomainPreprocessor
)

# ============================================================================
# TEST 1: STRUCTURE PRESERVATION
# ============================================================================
print("=" * 80)
print("TEST 1: STRUCTURE PRESERVATION")
print("=" * 80)

test_text = """• पहला बिंदु
  • उप-बिंदु
• दूसरा बिंदु

अनुच्छेद दो यहाँ है।"""

sp = StructurePreserver()
structure = sp.extract_structure(test_text)
print(f"Extracted {len(structure['paragraphs'])} paragraphs:")
for i, para in enumerate(structure['paragraphs']):
    print(f"  Para {i}: type={para.get('starts_with_bullet')}, indent={para.get('indent')}")

# ============================================================================
# TEST 2: NAMED ENTITY TRACKING
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: NAMED ENTITY TRACKING")
print("=" * 80)

ne_tracker = NamedEntityTracker()
text1 = "James Clear ने Atomic Habits लिखी।"
nes = ne_tracker.extract_named_entities(text1, "hin_Deva")
print(f"Extracted NEs: {[ne['text'] for ne in nes]}")

ne_tracker.update_mapping("James", "जेम्स", "tel_Telu")
print(f"Cached mapping: James → जेम्स (Telugu)")

# ============================================================================
# TEST 3: CONTEXT CACHE
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: CONTEXT-AWARE CACHE")
print("=" * 80)

cache = ContextAwareCache(max_cache_size=100)
cache.set("hello", "नमस्ते", "eng_Latn", "hin_Deva")
cached = cache.get("hello", "eng_Latn", "hin_Deva")
print(f"Cached translation retrieved: {cached}")

# ============================================================================
# TEST 4: QUALITY METRICS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: QUALITY METRICS")
print("=" * 80)

src = "This is a test sentence."
tgt = "यह एक परीक्षण वाक्य है।"
metrics = QualityMetrics.compute_metrics(src, tgt)
print(f"Quality Report:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ============================================================================
# TEST 5: ERROR DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: ERROR DETECTION")
print("=" * 80)

src = "Sentence 1. Sentence 2. Sentence 3."
tgt = "वाक्य 1।"  # Only 1 sentence (dropped)
errors = ErrorDetector.detect_all_errors(src, tgt)
print(f"Error Report:")
for err_type, err_info in errors.items():
    print(f"  {err_type}: {err_info['detected']} - {err_info['message']}")

# ============================================================================
# TEST 6: DOMAIN DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: DOMAIN DETECTION")
print("=" * 80)

legal_text = "Article 1 of the Constitution states that hereby..."
domain = DomainPreprocessor.detect_domain(legal_text)
print(f"Detected domain: {domain}")

# ============================================================================
# FULL PIPELINE TEST (with actual IndicTrans2 if available)
# ============================================================================
print("\n" + "=" * 80)
print("FULL PIPELINE TEST")
print("=" * 80)

try:
    # Load model (might fail on local setup without GPU)
    print("Loading IndicTrans2 model...")
    MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    ip = IndicProcessor(inference=True)

    # Sample text
    hindi_text = "नमस्ते। आज का दिन बहुत अच्छा है।"

    print(f"Source (Hindi): {hindi_text}")

    # Process through pipeline with ALL novelties
    src_tag = "hin_Deva"
    tgt_tag = "tel_Telu"

    # This is simplified; see enhanced_streamlit_app.py for full pipeline
    preprocessed = ip.preprocess_batch([hindi_text], src_lang=src_tag, tgt_lang=tgt_tag)
    tokenized = tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, num_beams=5, max_length=256)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    postprocessed = ip.postprocess_batch(decoded, lang=tgt_tag)

    print(f"Target (Telugu): {postprocessed[0] if postprocessed else decoded[0]}")

    # Apply novelties
    structure = StructurePreserver.extract_structure(hindi_text)
    print(f"✓ Structure extraction: {len(structure['paragraphs'])} segments")

    ne_tracker = NamedEntityTracker()
    print(f"✓ NE consistency module ready")

    metrics = QualityMetrics.compute_metrics(hindi_text, postprocessed[0], tokenizer)
    print(f"✓ Quality score: {metrics['avg_quality']:.2f}")

    errors = ErrorDetector.detect_all_errors(hindi_text, postprocessed[0])
    print(f"✓ Error detection: {sum(1 for e in errors.values() if e['detected'])} issues found")

except Exception as e:
    print(f"Note: Full pipeline requires IndicTrans2 setup: {e}")

print("\n" + "=" * 80)
print("QUICK START COMPLETE!")
print("=" * 80)
print("Next: Run 'streamlit run enhanced_streamlit_app.py' to launch web interface")
