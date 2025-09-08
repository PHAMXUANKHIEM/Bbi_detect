import warnings

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ArcFace:
    def __init__(self, device="cpu"):
        """
        Initialize ArcFace model for face embedding extraction.

        Args:
            device (str): Device to run the model ('cpu' or 'cuda').
        """
        self.device = device
        self.app = None

        try:
            # Initialize with proper providers based on device
            if device == "cuda" and torch.cuda.is_available():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("🔄 Initializing ArcFace with CUDA...")
            else:
                providers = ["CPUExecutionProvider"]
                print("🔄 Initializing ArcFace with CPU...")

            self.app = FaceAnalysis(
                name="buffalo_sc", providers=providers, allowed_modules=["detection", "recognition"]
            )

            # Prepare with appropriate context and detection size
            ctx_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            print(f"✅ ArcFace model initialized successfully on {device}")

        except Exception as e:
            print(f"❌ Error initializing ArcFace model: {e}")
            print("📝 Make sure you have installed insightface: pip install insightface")
            self.app = None

    def preprocess_image(self, image):
        """
        Preprocess image for ArcFace.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            numpy.ndarray: Preprocessed image or None if invalid.
        """
        try:
            # Validate image
            if image is None:
                print("❌ Image is None")
                return None

            if len(image.shape) != 3:
                print(f"❌ Invalid image dimensions: {image.shape}")
                return None

            if image.shape[2] != 3:
                print(f"❌ Image must have 3 channels, got: {image.shape[2]}")
                return None

            if image.size == 0 or image.shape[0] < 20 or image.shape[1] < 20:
                print(f"❌ Image too small: {image.shape}")
                return None

            # Ensure image is in correct format (BGR for OpenCV/InsightFace)
            # The input is RGB, so we need to convert to BGR for InsightFace
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Convert RGB to BGR (InsightFace expects BGR)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image_bgr

        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None

    def get_face_embedding(self, image):
        """
        Extract face embedding from image.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            numpy.ndarray: Face embedding or None if extraction fails.
        """
        if self.app is None:
            print("❌ ArcFace model not initialized")
            return None

        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return None

            # Detect faces and get embeddings
            faces = self.app.get(processed_img)

            if len(faces) == 0:
                # Don't print this as it's common and clutters output
                return None

            # Get embedding of the first (largest) face
            face = faces[0]  # InsightFace returns faces sorted by size
            embedding = face.embedding

            # Validate embedding
            if embedding is None:
                print("❌ Failed to extract embedding")
                return None

            if not isinstance(embedding, np.ndarray):
                print(f"❌ Invalid embedding type: {type(embedding)}")
                return None

            if embedding.size == 0:
                print("❌ Empty embedding")
                return None

            # Normalize embedding (important for cosine similarity)
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None

    def get_all_embeddings(self, image):
        """
        Extract embeddings for all faces in the image.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            list: List of face embeddings or empty list if extraction fails.
        """
        if self.app is None:
            print("❌ ArcFace model not initialized")
            return []

        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return []

            # Detect faces and get embeddings
            faces = self.app.get(processed_img)

            if len(faces) == 0:
                return []

            embeddings = []
            for face in faces:
                embedding = face.embedding
                if embedding is not None and embedding.size > 0:
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding.astype(np.float32))

            return embeddings

        except Exception as e:
            print(f"❌ Error extracting embeddings: {e}")
            return []

    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        """
        Compare two face embeddings using cosine similarity.

        Args:
            embedding1 (numpy.ndarray): First face embedding.
            embedding2 (numpy.ndarray): Second face embedding.
            threshold (float): Similarity threshold (0.6 = 60% similar).

        Returns:
            tuple: (is_same_person, similarity_score)
        """
        try:
            if embedding1 is None or embedding2 is None:
                return False, 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)

            # Convert to distance (1 - similarity) for threshold comparison
            distance = 1 - similarity
            is_same = distance < (1 - threshold)

            return is_same, float(similarity)

        except Exception as e:
            print(f"❌ Error comparing faces: {e}")
            return False, 0.0

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "app") and self.app is not None:
                del self.app
        except:
            pass
