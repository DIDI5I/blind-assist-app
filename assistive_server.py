from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import ollama
import io
import logging
from datetime import datetime
import concurrent.futures
from collections import defaultdict
from werkzeug.serving import WSGIRequestHandler
import socket

# ======================================================
# ============ CONFIGURATION SECTION ===================
# ======================================================
DETECTION_CONFIDENCE = 0.5
MAX_IMAGE_DIMENSION = 1280
LLM_TIMEOUT = 60  # seconds
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
MODEL_NAME = "phi3:mini"
THREAD_POOL_WORKERS = 4

TESTING_MODE = True
LOCAL_IP = "127.0.0.1"
EXTERNAL_IP = "192.168.1.5"

# ======================================================
# ============ INITIALIZATION ==========================
# ======================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

model = YOLO("yolov8n.pt")
model.overrides = {
    'conf': DETECTION_CONFIDENCE,
    'iou': 0.45,
    'agnostic_nms': True,
    'imgsz': 640
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

# ======================================================
# ============ MEMORY STATE FOR LLM ====================
# ======================================================

# Persistent chat session state
chat_history = []

def generate_description(detections):
    """Generate a natural language description using a persistent LLM chat"""
    if not detections:
        return "I can't confidently identify any objects."

    object_counts = defaultdict(int)
    for detection in detections:
        object_counts[detection['class']] += 1

    items = [f"{count} {obj}" + ("" if count == 1 else "s") for obj, count in object_counts.items()]
    objects_text = ', '.join(items[:-1]) + (f" and {items[-1]}" if len(items) > 1 else items[0])
    prompt = f"Briefly(<50 WORDS) in a descriptive manner provide context of the scene : {objects_text}"

    chat_history.append({"role": "user", "content": prompt})

    try:
        future = executor.submit(
            ollama.chat,
            model=MODEL_NAME,
            messages=chat_history,
            options={'temperature': 0.2, 'num_ctx': 256}
        )
        response = future.result(timeout=LLM_TIMEOUT)
        reply = response['message']['content'].strip()
        chat_history.append({"role": "assistant", "content": reply})
        return reply.split('.')[0] + '.' if '.' in reply else reply
    except concurrent.futures.TimeoutError:
        logger.warning("LLM timeout, returning basic description")
        return f"I see {objects_text}."
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return f"I see {objects_text}."

# ======================================================
# ============ FLASK ROUTES ============================
# ======================================================

@app.route('/process-image', methods=['POST'])
def process_image():
    start_time = datetime.now()

    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded",
            "processing_time": 0
        }), 400

    try:
        img_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(img_bytes))

        # Resize image
        ratio = min(MAX_IMAGE_DIMENSION / img.width, MAX_IMAGE_DIMENSION / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Object detection
        results = model.predict(img, verbose=False)
        detections = [{
            'class': model.names[int(box.cls)],
            'confidence': float(box.conf),
            'bbox': box.xyxy.tolist()[0]
        } for result in results for box in result.boxes]

        # Scene description
        description = generate_description(detections)

        return jsonify({
            "status": "success",
            "description": description,
            "detections": detections,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "image_resolution": new_size
        })

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Image processing failed",
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "server_time": datetime.now().isoformat()
    })

# ======================================================
# ============ ENTRY POINT =============================
# ======================================================

if __name__ == "__main__":
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    socket.setdefaulttimeout(40)

    # Warm up models
    dummy_img = Image.new('RGB', (640, 640))
    model.predict(dummy_img, verbose=False)
    ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "Hello"}])

    bind_ip = LOCAL_IP if TESTING_MODE else EXTERNAL_IP

    print(f"🔧 Running Flask server on http://{bind_ip}:8000")
    print(f"🌐 (Switch TESTING_MODE to False for Android phone access)")

    app.run(
        host=bind_ip,
        port=8000,
        threaded=True,
        debug=False,
        use_reloader=False
    )
