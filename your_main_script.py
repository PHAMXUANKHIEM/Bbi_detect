# ============ KEY FIXES FOR YOUR MAIN SCRIPT ============
from mediapipe.tasks.python.vision import face_detector

from test import FACE_PROCESS_INTERVAL


# 1. UPDATE THE FACE EMBEDDING FUNCTION
def get_face_embedding(face_img, arcface_model):
    """
    Get face embedding using ArcFace model.

    Args:
        face_img (numpy.ndarray): Face image in BGR format (from OpenCV)
        arcface_model: ArcFace model instance

    Returns:
        numpy.ndarray: Face embedding or None if failed
    """
    try:
        if face_img is None or face_img.size == 0:
            return None

        # Convert BGR to RGB for ArcFace (it expects RGB)
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Get embedding
        embedding = arcface_model.get_embedding(face_img_rgb)

        return embedding

    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None


# 2. UPDATE THE STRANGER DETECTION FUNCTION
def is_stranger(embedding, known_encodings, threshold=0.6):
    """
    Check if a face embedding belongs to a stranger.

    Args:
        embedding (numpy.ndarray): Face embedding to check
        known_encodings (list): List of known face embeddings
        threshold (float): Similarity threshold (0.6 = need 60% similarity to be known)

    Returns:
        bool: True if stranger, False if known person
    """
    if embedding is None or len(known_encodings) == 0:
        return True

    try:
        max_similarity = 0.0

        for known_encoding in known_encodings:
            if known_encoding is not None:
                # Calculate cosine similarity (normalized dot product)
                similarity = np.dot(embedding, known_encoding)
                max_similarity = max(max_similarity, similarity)

        # If highest similarity is below threshold, it's a stranger
        return max_similarity < threshold

    except Exception as e:
        print(f"❌ Error computing similarity: {e}")
        return True


# 3. UPDATE THE ARCFACE INITIALIZATION IN YOUR MAIN SCRIPT
print("🔄 Loading ArcFace model...")
try:
    # Use CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arcface_model = ArcFace(device=device)

    if arcface_model.app is not None:
        print(f"✅ ArcFace model loaded successfully on {device}")
    else:
        print("❌ ArcFace model failed to initialize")
        arcface_model = None

except Exception as e:
    print(f"❌ Error loading ArcFace model: {e}")
    arcface_model = None

# 4. UPDATE THE FACE PROCESSING SECTION IN YOUR MAIN LOOP
if arcface_model and arcface_model.app and frame_count % FACE_PROCESS_INTERVAL == 0:
    faces = face_detector.detect_faces(frame)

    for (x, y, w, h) in faces:
        # Skip faces that are inside baby detection boxes (likely baby faces)
        if is_child_face((x, y, w, h), baby_boxes):
            continue

        # Extract face region with some padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        # Get face embedding
        embedding = get_face_embedding(face_img, arcface_model)

        if embedding is None:
            continue

        # Check if stranger
        stranger = is_stranger(embedding, known_encodings, STRANGER_DISTANCE_THRESHOLD)

        # Draw bounding box and label
        color = (0, 0, 255) if stranger else (0, 255, 0)  # Red for stranger, Green for known
        label = "Nguoi_la" if stranger else (known_names[0] if known_names else "Da_xac_dinh")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Alert for strangers
        if stranger and current_time - last_alert_time['face'] > alert_interval:
            print(f"🚨 CẢNH BÁO: Phát hiện người lạ! ({datetime.now().strftime('%H:%M:%S')})")
            last_alert_time['face'] = current_time

            # Save snapshot
            snapshot_path = f"snapshots/stranger_{int(current_time)}.jpg"
            os.makedirs("snapshots", exist_ok=True)
            cv2.imwrite(snapshot_path, frame)

            try:
                upload_to_drive(snapshot_path, GOOGLE_DRIVE_FOLDER_ID)
                print(f"📤 Snapshot uploaded: {snapshot_path}")
            except Exception as e:
                print(f"❌ Failed to upload snapshot: {e}")


# 5. RECOMMENDED: ADD ERROR HANDLING FOR ARCFACE MODEL FAILURES
def safe_arcface_operation(func, *args, **kwargs):
    """
    Safely execute ArcFace operations with error handling.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ ArcFace operation failed: {e}")
        return None