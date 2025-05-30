import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN

class FaceRecognitionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.feature_extractor = None
        self.face_detector = MTCNN()
        
    def build_model(self):
        """Build enhanced CNN model for face recognition with improved feature extraction"""
        # Use ResNet50V2 as base model with fine-tuned parameters
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=self.input_shape, pooling=None)
        
        # Freeze early layers to preserve low-level features
        for layer in base_model.layers[:100]:  # Freeze first 100 layers
            layer.trainable = False
        
        # Create a more sophisticated feature extraction pipeline
        x = base_model.output
        
        # Add spatial attention mechanism
        attention = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(1024, activation='relu')(attention)
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
        attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
        x = layers.Multiply()([x, attention])
        
        # Add global pooling with both average and max pooling for better feature capture
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Add feature normalization for better embedding quality
        x = layers.BatchNormalization()(x)
        
        # Add deeper feature extraction layers
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)  # Slightly reduced dropout for better feature preservation
        
        # Add a bottleneck layer for more compact and discriminative embeddings
        embedding = layers.Dense(512, activation=None)(x)  # Linear activation for raw embeddings
        normalized_embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)
        
        # Create feature extractor model with normalized embeddings
        self.feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=normalized_embedding)
        
        # Create classification model if num_classes is provided
        if self.num_classes:
            # Add classification head
            output = layers.Dense(self.num_classes, activation='softmax')(embedding)
            
            # Create full model
            model = tf.keras.Model(inputs=base_model.input, outputs=output)
            self.model = model
            return model
            
        return None
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate optimizer and loss function"""
        if self.model is None:
            raise ValueError("Model must be built before compiling")
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_dir, validation_dir, epochs=20, batch_size=32):
        """Train the model using enhanced data augmentation and training strategies"""
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Enhanced data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            # Basic preprocessing
            rescale=1./255,
            # Standardization for better convergence
            featurewise_center=True,
            featurewise_std_normalization=True,
            # Geometric transformations
            rotation_range=30,  # Increased rotation range
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,  # Increased shear for more variation
            zoom_range=[0.8, 1.2],  # Asymmetric zoom range
            horizontal_flip=True,
            # Color augmentations for lighting invariance
            brightness_range=[0.7, 1.3],  # Brightness variation
            channel_shift_range=20.0,  # Color channel shifts
            # Fill mode for transformations
            fill_mode='reflect',  # Better edge handling
            # Additional preprocessing function
            preprocessing_function=self._random_augmentation
        )
        
        # Compute statistics for standardization
        # This requires loading a subset of images to compute mean and std
        try:
            # Load a small batch of images to compute statistics
            temp_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=min(500, batch_size*10),  # Use a reasonable number of images
                class_mode='categorical',
                shuffle=True
            )
            # Get a batch of images
            batch_x, _ = next(temp_gen)
            # Compute statistics
            train_datagen.fit(batch_x)
        except Exception as e:
            print(f"Warning: Could not compute dataset statistics: {str(e)}")
            # Fallback to simpler augmentation without standardization
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.3,
                zoom_range=[0.8, 1.2],
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                fill_mode='reflect',
                preprocessing_function=self._random_augmentation
            )
        
        # Validation data generator with minimal augmentation
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            # Apply same standardization as training if available
            featurewise_center=train_datagen.featurewise_center,
            featurewise_std_normalization=train_datagen.featurewise_std_normalization,
            # Copy mean and std from training generator
            mean=train_datagen.mean if hasattr(train_datagen, 'mean') else None,
            std=train_datagen.std if hasattr(train_datagen, 'std') else None
        )
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False  # No need to shuffle validation data
        )
        
        # Create callbacks for better training
        callbacks = [
            # Learning rate scheduler
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_face_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with improved settings
        history = self.model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=max(1, validation_generator.samples // batch_size),
            callbacks=callbacks,
            workers=4,  # Parallel processing
            use_multiprocessing=True,
            shuffle=True
        )
        
        # Load the best weights after training
        try:
            if os.path.exists('best_face_model.h5'):
                self.model.load_weights('best_face_model.h5')
                print("Loaded best model weights from checkpoint")
        except Exception as e:
            print(f"Could not load best model weights: {str(e)}")
        
        return history
        
    def _random_augmentation(self, image):
        """Apply random augmentations to further enhance training data diversity"""
        # Skip augmentation randomly to maintain some original samples
        if np.random.random() < 0.3:  # 30% chance to skip augmentation
            return image
            
        # Apply random noise (occasionally)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
            
        # Apply random blur (occasionally)
        if np.random.random() < 0.2:
            blur_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
            
        # Apply random contrast/brightness adjustment
        if np.random.random() < 0.3:
            alpha = 1.0 + np.random.uniform(-0.3, 0.3)  # Contrast control
            beta = np.random.uniform(-0.1, 0.1)  # Brightness control
            image = np.clip(alpha * image + beta, 0, 1)
            
        # Apply random occlusion (simulate partial face coverage)
        if np.random.random() < 0.15:
            h, w, _ = image.shape
            occlusion_size_h = int(h * np.random.uniform(0.1, 0.3))
            occlusion_size_w = int(w * np.random.uniform(0.1, 0.3))
            x = np.random.randint(0, w - occlusion_size_w)
            y = np.random.randint(0, h - occlusion_size_h)
            image[y:y+occlusion_size_h, x:x+occlusion_size_w, :] = np.random.random(3)
            
        return image
    
    def predict(self, image):
        """Predict the class of a single image"""
        if self.model is None:
            raise ValueError("Model must be built before prediction")
            
        if isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        image = self.preprocess_image(image)
        if image is None:
            return None
            
        # Get predictions
        predictions = self.model.predict(np.expand_dims(image, axis=0))
        return predictions
    
    def extract_features(self, image):
        """Extract features from a face image using the enhanced CNN model with improved normalization"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor must be built before extraction")
            
        # Ensure image is in correct format
        if image is None:
            return None
        
        try:
            # Check image quality before feature extraction
            if isinstance(image, np.ndarray):
                # Check for extremely low contrast or brightness issues
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.dtype == np.float32 else cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    
                    # Calculate image statistics
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    
                    # Apply additional preprocessing for problematic images
                    if contrast < 10 or brightness < 30 or brightness > 225:
                        # Apply additional contrast enhancement
                        if image.dtype == np.float32:
                            # For normalized images (0-1)
                            min_val = np.min(image)
                            max_val = np.max(image)
                            if max_val > min_val:  # Avoid division by zero
                                image = (image - min_val) / (max_val - min_val)
                        else:
                            # For uint8 images
                            image = cv2.equalizeHist(gray)
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                            image = image.astype('float32') / 255.0
            
            # Expand dimensions for batch processing
            image_batch = np.expand_dims(image, axis=0)
            
            # Extract features with error handling
            features = self.feature_extractor.predict(image_batch, verbose=0)  # Suppress verbose output
            
            # Apply additional L2 normalization for more consistent similarity calculation
            features_normalized = tf.math.l2_normalize(features, axis=1).numpy()
            
            return features_normalized[0]  # Return the normalized feature vector
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
    
    # Class-level cache for student embeddings to improve performance
    _embedding_cache = {}
    
    def extract_features_batch(self, images):
        """Extract features from a batch of face images with enhanced parallel processing and caching"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor must be built before extraction")
            
        # Preprocess all images with progress tracking
        processed_images = []
        valid_indices = []
        cache_hits = 0
        
        # Check cache first for each image
        for i, img in enumerate(images):
            try:
                # Generate a cache key based on image content
                if isinstance(img, np.ndarray):
                    # Use a hash of the image data as cache key
                    # Downsample to reduce sensitivity to minor pixel changes
                    small_img = cv2.resize(img, (32, 32))
                    cache_key = hash(small_img.tobytes())
                    
                    # Check if we have this image in cache
                    if cache_key in self._embedding_cache:
                        # Use cached embedding
                        cache_hits += 1
                        continue
                
                # Apply enhanced preprocessing if not in cache
                processed = self.preprocess_image(img)
                if processed is not None:
                    processed_images.append(processed)
                    valid_indices.append(i)
                    # Store the cache key for later
                    if isinstance(img, np.ndarray):
                        processed_images[-1].cache_key = cache_key
            except Exception as e:
                print(f"Error preprocessing image {i}: {str(e)}")
                continue
        
        # If all images were in cache, return cached results
        if cache_hits == len(images):
            # Return cached embeddings
            cached_embeddings = []
            cached_indices = []
            
            for i, img in enumerate(images):
                if isinstance(img, np.ndarray):
                    small_img = cv2.resize(img, (32, 32))
                    cache_key = hash(small_img.tobytes())
                    if cache_key in self._embedding_cache:
                        cached_embeddings.append(self._embedding_cache[cache_key])
                        cached_indices.append(i)
            
            if cached_embeddings:
                return np.array(cached_embeddings), cached_indices
        
        if not processed_images:
            return None, []
            
        try:
            # Convert to batch with proper error handling
            image_batch = np.array(processed_images)
            
            # Process in smaller batches to avoid memory issues
            batch_size = 32  # Process 32 images at a time
            all_features = []
            
            for i in range(0, len(image_batch), batch_size):
                batch_slice = image_batch[i:i+batch_size]
                # Extract features
                batch_features = self.feature_extractor.predict(batch_slice, verbose=0)  # Suppress verbose output
                # Apply L2 normalization for consistent similarity calculation
                batch_features = tf.math.l2_normalize(batch_features, axis=1).numpy()
                all_features.append(batch_features)
            
            # Combine all batches
            if all_features:
                features = np.vstack(all_features)
                
                # Update cache with new embeddings
                for i, idx in enumerate(valid_indices):
                    img = images[idx]
                    if isinstance(img, np.ndarray) and hasattr(processed_images[i], 'cache_key'):
                        cache_key = processed_images[i].cache_key
                        self._embedding_cache[cache_key] = features[i]
                
                # Limit cache size to prevent memory issues (keep most recent 1000 entries)
                if len(self._embedding_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self._embedding_cache.keys())[:-1000]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
                
                return features, valid_indices
            return None, []
            
        except Exception as e:
            print(f"Batch feature extraction error: {str(e)}")
            return None, []
    
    def preprocess_image(self, image, detect_face=True):
        """Preprocess image for the model with enhanced face detection and alignment"""
        if image is None:
            return None
            
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                # Check if image is BGR (OpenCV default)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            # Convert grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply advanced histogram equalization to improve contrast in grayscale domain
        gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Use CLAHE instead of simple histogram equalization for better contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray_img)
        
        # Convert back to RGB for further processing
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        
        # Detect and crop face if requested
        if detect_face:
            try:
                # Detect faces with both original and equalized images for better detection
                # Use a lower confidence threshold to detect more faces in challenging conditions
                faces_original = self.face_detector.detect_faces(image_rgb)
                faces_equalized = self.face_detector.detect_faces(equalized_rgb)
                
                # Combine face detections, prioritizing higher confidence
                faces = faces_original + faces_equalized
                
                if not faces:
                    # Apply multiple preprocessing techniques to enhance face detection
                    # 1. Gaussian blur to reduce noise
                    blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)
                    faces_blurred = self.face_detector.detect_faces(blurred)
                    
                    # 2. Increase contrast
                    contrast_enhanced = cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=0)
                    faces_contrast = self.face_detector.detect_faces(contrast_enhanced)
                    
                    # Combine all detection attempts
                    faces = faces_blurred + faces_contrast
                    
                    if not faces:
                        return None
                
                # Get the face with highest confidence
                face = max(faces, key=lambda x: x['confidence'])
                if face['confidence'] < 0.8:  # More permissive threshold for challenging conditions
                    return None
                
                # Extract facial landmarks for alignment
                landmarks = face.get('keypoints', None)
                
                # Extract face with adaptive padding based on face size and position
                x, y, w, h = face['box']
                face_size = max(w, h)
                
                # Adaptive padding based on face size and position in the image
                # More padding for faces near the edge of the image
                edge_distance = min(x, y, image_rgb.shape[1]-x-w, image_rgb.shape[0]-y-h)
                edge_factor = max(0.1, min(0.3, 1.0 - (edge_distance / face_size)))
                
                # Base padding ratio depends on face size
                base_padding_ratio = 0.35 if face_size < 100 else 0.25
                padding_ratio = base_padding_ratio + edge_factor
                
                padding = int(padding_ratio * face_size)
                
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image_rgb.shape[1], x + w + padding)
                y_end = min(image_rgb.shape[0], y + h + padding)
                face_img = image_rgb[y_start:y_end, x_start:x_end]
                
                # Apply face alignment if landmarks are available
                if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    
                    # Calculate angle for alignment
                    dx = right_eye[0] - left_eye[0]
                    dy = right_eye[1] - left_eye[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Adjust coordinates for the cropped face image
                    left_eye = (left_eye[0] - x_start, left_eye[1] - y_start)
                    right_eye = (right_eye[0] - x_start, right_eye[1] - y_start)
                    
                    # Calculate center of the face for rotation with robust type checking and error handling
                    try:
                        # Ensure coordinates are valid numbers
                        if not isinstance(left_eye, tuple) or not isinstance(right_eye, tuple):
                            raise ValueError(f"Eye coordinates must be tuples, got {type(left_eye)} and {type(right_eye)}")
                            
                        if len(left_eye) != 2 or len(right_eye) != 2:
                            raise ValueError(f"Eye coordinates must have exactly 2 values, got {len(left_eye)} and {len(right_eye)}")
                        
                        # Convert to float first to handle any numeric type, then to int for pixel coordinates
                        try:
                            left_eye_x, left_eye_y = float(left_eye[0]), float(left_eye[1])
                            right_eye_x, right_eye_y = float(right_eye[0]), float(right_eye[1])
                        except (TypeError, ValueError) as e:
                            raise ValueError(f"Cannot convert eye coordinates to numbers: {e}")
                        
                        # Calculate center point using float division for accuracy, then convert to int for pixels
                        center_x = int((left_eye_x + right_eye_x) / 2)
                        center_y = int((left_eye_y + right_eye_y) / 2)
                        center = (center_x, center_y)
                        
                        # Validate center coordinates are within image bounds
                        if center_x < 0 or center_x >= face_img.shape[1] or center_y < 0 or center_y >= face_img.shape[0]:
                            raise ValueError(f"Center coordinates {center} are outside image bounds {face_img.shape}")
                        
                        # Get rotation matrix and apply rotation
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        face_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), 
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    except Exception as e:
                        print(f"Face alignment error: {str(e)}. Skipping alignment.")
                        # Continue without alignment if there's an error
                
                # Resize to target size with better interpolation
                face_img = cv2.resize(face_img, self.input_shape[:2], interpolation=cv2.INTER_LANCZOS4)
                
                # Apply advanced color correction and lighting normalization
                # 1. Convert to LAB color space for better color processing
                lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # 2. Apply CLAHE with adaptive parameters based on image brightness
                avg_brightness = np.mean(l)
                clip_limit = 3.0 if avg_brightness < 100 or avg_brightness > 200 else 2.0
                
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # 3. Apply gamma correction for better dynamic range
                # Darker images get gamma < 1, brighter images get gamma > 1
                gamma = 0.8 if avg_brightness < 100 else (1.2 if avg_brightness > 200 else 1.0)
                l_gamma = np.power(l / 255.0, gamma) * 255.0
                l_gamma = l_gamma.astype(np.uint8)
                
                # 4. Merge channels and convert back to RGB
                lab_enhanced = cv2.merge((l_gamma, a, b))
                face_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
            except Exception as e:
                print(f"Face detection error: {str(e)}")
                # Fallback to using the whole image with enhanced preprocessing
                face_img = cv2.resize(image_rgb, self.input_shape[:2])
                
                # Apply advanced lighting normalization for the fallback case
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                avg_brightness = np.mean(gray)
                
                # Apply adaptive CLAHE based on image brightness
                clip_limit = 3.0 if avg_brightness < 100 or avg_brightness > 200 else 2.0
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                
                # Apply to each channel separately for better color preservation
                r, g, b = cv2.split(face_img)
                r = clahe.apply(r)
                g = clahe.apply(g)
                b = clahe.apply(b)
                face_img = cv2.merge((r, g, b))
        else:
            # Just resize without face detection but with improved preprocessing
            face_img = cv2.resize(image_rgb, self.input_shape[:2], interpolation=cv2.INTER_LANCZOS4)
            
            # Apply advanced lighting normalization
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            
            # Apply adaptive CLAHE based on image brightness
            clip_limit = 3.0 if avg_brightness < 100 or avg_brightness > 200 else 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            
            # Apply to each channel separately for better color preservation
            r, g, b = cv2.split(face_img)
            r = clahe.apply(r)
            g = clahe.apply(g)
            b = clahe.apply(b)
            face_img = cv2.merge((r, g, b))
        
        # Normalize pixel values with improved scaling
        face_img = face_img.astype('float32') / 255.0
        
        # Apply standardization for better model performance
        # Use per-image standardization for better handling of lighting variations
        mean = np.mean(face_img, axis=(0, 1), keepdims=True)
        std = np.std(face_img, axis=(0, 1), keepdims=True) + 1e-7  # Avoid division by zero
        face_img = (face_img - mean) / std
        
        return face_img
    
    def find_best_match(self, face_embedding, student_embeddings, student_ids):
        """Find the best matching student for a given face embedding with enhanced matching algorithm"""
        if face_embedding is None or student_embeddings is None or len(student_embeddings) == 0:
            return None, 0
            
        # Reshape face embedding for comparison
        face_embedding = face_embedding.reshape(1, -1)
        
        # Normalize embeddings for more consistent similarity calculation
        face_embedding_norm = tf.math.l2_normalize(face_embedding, axis=1).numpy()
        student_embeddings_norm = tf.math.l2_normalize(student_embeddings, axis=1).numpy()
        
        # Calculate multiple similarity metrics for more robust matching
        # 1. Cosine similarity (primary metric)
        cosine_similarities = cosine_similarity(face_embedding_norm, student_embeddings_norm)
        
        # 2. Euclidean distance (secondary metric, convert to similarity where higher is better)
        euclidean_distances = np.zeros(len(student_embeddings))
        for i, emb in enumerate(student_embeddings):
            euclidean_distances[i] = np.linalg.norm(face_embedding.flatten() - emb)
        # Convert distances to similarities (1 / (1 + distance))
        euclidean_similarities = 1.0 / (1.0 + euclidean_distances)
        
        # 3. Manhattan distance as a third metric (more robust to outliers)
        manhattan_distances = np.zeros(len(student_embeddings))
        for i, emb in enumerate(student_embeddings):
            manhattan_distances[i] = np.sum(np.abs(face_embedding.flatten() - emb))
        # Convert to similarities
        manhattan_similarities = 1.0 / (1.0 + manhattan_distances)
        
        # 4. Combine similarity scores with weighted average
        # Prioritize cosine similarity (70%) but consider euclidean (20%) and manhattan (10%)
        combined_similarities = 0.7 * cosine_similarities[0] + 0.2 * euclidean_similarities + 0.1 * manhattan_similarities
        
        # Find top 5 matches for ensemble decision (increased from 3 for better coverage)
        top_indices = np.argsort(-combined_similarities)[:5]  # Negative for descending order
        top_similarities = combined_similarities[top_indices]
        
        # Check if the best match is significantly better than the second best
        if len(top_indices) > 1 and (top_similarities[0] - top_similarities[1]) > 0.08:  # Reduced threshold
            # Clear winner - use the top match
            best_match_idx = top_indices[0]
            best_match_similarity = top_similarities[0]
        else:
            # Close matches - use weighted voting among top matches
            # Weight votes by their similarity scores
            votes = np.zeros(len(student_embeddings))
            for i, idx in enumerate(top_indices[:3]):  # Consider top 3 for voting
                weight = max(0, top_similarities[i])  # Ensure positive weight
                votes[idx] += weight
            
            # Select the candidate with highest weighted votes
            best_match_idx = np.argmax(votes)
            best_match_similarity = combined_similarities[best_match_idx]
        
        # Return the student ID and similarity score
        return student_ids[best_match_idx], best_match_similarity
    
    def compare_faces(self, face_image1, face_image2, threshold=0.6, adaptive_threshold=True):
        """Compare two face images with enhanced similarity calculation and adaptive thresholding"""
        # Extract features from both images with enhanced preprocessing
        embedding1 = self.extract_features(self.preprocess_image(face_image1))
        embedding2 = self.extract_features(self.preprocess_image(face_image2))
        
        if embedding1 is None or embedding2 is None:
            return False, 0.0, {"error": "Could not extract features from one or both images"}
        
        # Calculate similarity using multiple metrics for more robust comparison
        # Cosine similarity (primary metric)
        cosine_sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        
        # Euclidean distance (secondary metric, lower is better)
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        # Convert to similarity score (higher is better)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine metrics with weighted average (prioritize cosine similarity)
        similarity = 0.8 * cosine_sim + 0.2 * euclidean_sim
        
        # Calculate image quality metrics for adaptive thresholding
        metrics = {}
        
        if adaptive_threshold:
            # Adjust threshold based on image quality and other factors
            adjusted_threshold = threshold
            
            try:
                # Check if we have the original images to analyze quality
                if isinstance(face_image1, np.ndarray) and isinstance(face_image2, np.ndarray):
                    # Calculate brightness and contrast metrics
                    img1_gray = cv2.cvtColor(face_image1, cv2.COLOR_BGR2GRAY) if len(face_image1.shape) == 3 else face_image1
                    img2_gray = cv2.cvtColor(face_image2, cv2.COLOR_BGR2GRAY) if len(face_image2.shape) == 3 else face_image2
                    
                    # Calculate average brightness
                    brightness1 = np.mean(img1_gray)
                    brightness2 = np.mean(img2_gray)
                    brightness_diff = abs(brightness1 - brightness2) / 255.0
                    
                    # Calculate contrast using standard deviation
                    contrast1 = np.std(img1_gray)
                    contrast2 = np.std(img2_gray)
                    contrast_diff = abs(contrast1 - contrast2) / 128.0
                    
                    # Store metrics for debugging
                    metrics["brightness1"] = float(brightness1)
                    metrics["brightness2"] = float(brightness2)
                    metrics["brightness_diff"] = float(brightness_diff)
                    metrics["contrast1"] = float(contrast1)
                    metrics["contrast2"] = float(contrast2)
                    metrics["contrast_diff"] = float(contrast_diff)
                    
                    # Adjust threshold based on image quality differences
                    # If images have very different lighting conditions, be more lenient
                    if brightness_diff > 0.3 or contrast_diff > 0.3:
                        adjusted_threshold -= 0.05  # More permissive
                        metrics["threshold_adjustment"] = "Lighting difference detected, lowered threshold"
                    
                    # If both images have poor contrast, be more lenient
                    if contrast1 < 30 and contrast2 < 30:
                        adjusted_threshold -= 0.05  # More permissive
                        metrics["threshold_adjustment"] = "Low contrast detected, lowered threshold"
                    
                    # If both images have extreme brightness (too dark or too bright), be more lenient
                    if (brightness1 < 50 or brightness1 > 200) and (brightness2 < 50 or brightness2 > 200):
                        adjusted_threshold -= 0.05  # More permissive
                        metrics["threshold_adjustment"] = "Extreme brightness detected, lowered threshold"
                    
                    # Ensure threshold stays within reasonable bounds
                    adjusted_threshold = max(0.45, min(0.8, adjusted_threshold))
                    metrics["final_threshold"] = float(adjusted_threshold)
            except Exception as e:
                print(f"Error in adaptive thresholding: {str(e)}")
                # Fall back to default threshold
                adjusted_threshold = threshold
                metrics["threshold_adjustment"] = f"Error in adaptive thresholding: {str(e)}"
        else:
            # Use fixed threshold
            adjusted_threshold = threshold
            metrics["threshold_adjustment"] = "Using fixed threshold"
        
        # Store similarity scores in metrics
        metrics["cosine_similarity"] = float(cosine_sim)
        metrics["euclidean_similarity"] = float(euclidean_sim)
        metrics["combined_similarity"] = float(similarity)
        
        # Determine if same person based on adjusted threshold
        is_same_person = similarity >= adjusted_threshold
        
        return is_same_person, similarity, metrics
    
    def save_model(self, model_path):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(model_path)
    
    @classmethod
    def load_model(cls, model_path, input_shape=(224, 224, 3)):
        """Load a saved model from disk"""
        # Create a new instance
        instance = cls(input_shape=input_shape)
        
        # Load the model
        instance.model = tf.keras.models.load_model(model_path)
        
        # Extract the base model for feature extraction
        # Assuming the base model is the first part of the loaded model
        if isinstance(instance.model, tf.keras.Sequential):
            base_model = instance.model.layers[0]
            instance.feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
        
        return instance

def preprocess_face_image(image_path, target_size=(224, 224)):
    """Preprocess face image for the model"""
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image

def train_face_recognition_model(data_dir, model_save_path, epochs=20):
    """Train and save a face recognition model"""
    # Count number of classes (students)
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Initialize and build model
    face_model = FaceRecognitionModel(num_classes=num_classes)
    face_model.build_model()
    face_model.compile_model()
    
    # Split data into train and validation
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    
    # Train the model
    history = face_model.train(train_dir, validation_dir, epochs=epochs)
    
    # Save the model
    face_model.save_model(model_save_path)
    
    return history

def recognize_face_from_camera(camera_image, student_images, threshold=0.6):
    """Recognize a face from camera image by comparing with student images using enhanced recognition"""
    # Start timing for performance monitoring
    start_total = time.time()
    
    # Initialize model with enhanced architecture
    model = FaceRecognitionModel()
    model.build_model()
    
    # Process camera image with improved preprocessing
    preprocess_start = time.time()
    camera_face = model.preprocess_image(camera_image)
    preprocess_time = time.time() - preprocess_start
    print(f"Preprocessing time: {preprocess_time:.3f} seconds")
    
    if camera_face is None:
        return None, 0.0, "No face detected in camera image"
    
    # Extract features from camera face with enhanced feature extraction
    feature_start = time.time()
    camera_embedding = model.extract_features(camera_face)
    feature_time = time.time() - feature_start
    print(f"Feature extraction time: {feature_time:.3f} seconds")
    
    if camera_embedding is None:
        return None, 0.0, "Failed to extract features from camera image"
    
    # Process all student images in batch for efficiency with caching
    batch_start = time.time()
    student_embeddings, valid_indices = model.extract_features_batch(student_images)
    batch_time = time.time() - batch_start
    print(f"Batch processing time: {batch_time:.3f} seconds for {len(student_images)} images")
    
    if student_embeddings is None or len(student_embeddings) == 0:
        return None, 0.0, "No valid student faces for comparison"
    
    # Use the enhanced find_best_match function for more accurate matching
    match_start = time.time()
    student_ids = list(range(len(student_images)))
    mapped_ids = [student_ids[i] for i in valid_indices]
    
    # Find best match using improved algorithm
    best_match_id, best_similarity = model.find_best_match(camera_embedding, student_embeddings, mapped_ids)
    match_time = time.time() - match_start
    print(f"Matching time: {match_time:.3f} seconds")
    
    # Apply enhanced adaptive thresholding based on multiple factors
    threshold_start = time.time()
    
    # Start with base threshold
    adaptive_threshold = threshold
    threshold_factors = {}
    
    try:
        # Check if we can analyze the camera image quality
        if isinstance(camera_image, np.ndarray):
            # Convert to grayscale for analysis
            if len(camera_image.shape) == 3:
                gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = camera_image
                
            # Calculate comprehensive image statistics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()  # Measure blurriness
            
            # Store factors for logging
            threshold_factors['brightness'] = float(brightness)
            threshold_factors['contrast'] = float(contrast)
            threshold_factors['blur_score'] = float(blur_score)
            
            # 1. Adjust for brightness issues
            brightness_adjustment = 0
            if brightness < 40:  # Very dark
                brightness_adjustment = -0.08
            elif brightness < 60:  # Dark
                brightness_adjustment = -0.06
            elif brightness > 220:  # Very bright
                brightness_adjustment = -0.08
            elif brightness > 180:  # Bright
                brightness_adjustment = -0.06
            threshold_factors['brightness_adjustment'] = brightness_adjustment
            adaptive_threshold += brightness_adjustment
            
            # 2. Adjust for contrast issues
            contrast_adjustment = 0
            if contrast < 20:  # Very low contrast
                contrast_adjustment = -0.08
            elif contrast < 40:  # Low contrast
                contrast_adjustment = -0.05
            threshold_factors['contrast_adjustment'] = contrast_adjustment
            adaptive_threshold += contrast_adjustment
            
            # 3. Adjust for blur
            blur_adjustment = 0
            if blur_score < 50:  # Very blurry
                blur_adjustment = -0.07
            elif blur_score < 100:  # Somewhat blurry
                blur_adjustment = -0.04
            threshold_factors['blur_adjustment'] = blur_adjustment
            adaptive_threshold += blur_adjustment
            
            # 4. Time of day adjustment (lighting conditions vary by time)
            time_adjustment = 0
            current_hour = datetime.now().hour
            if current_hour < 7 or current_hour > 19:  # Early morning or night
                time_adjustment = -0.05
            elif current_hour < 9 or current_hour > 17:  # Morning or evening
                time_adjustment = -0.03
            threshold_factors['time_adjustment'] = time_adjustment
            adaptive_threshold += time_adjustment
            
            # 5. Student count factor (more students = slightly more lenient)
            count_adjustment = min(0.04, len(valid_students) * 0.002) * -1
            threshold_factors['count_adjustment'] = count_adjustment
            adaptive_threshold += count_adjustment
            
            # Ensure threshold stays in reasonable range
            adaptive_threshold = max(0.40, min(0.75, adaptive_threshold))
            threshold_factors['final_threshold'] = adaptive_threshold
    except Exception as e:
        print(f"Error in adaptive thresholding: {str(e)}")
        # Fall back to base threshold with a small safety margin
        adaptive_threshold = threshold - 0.05
        threshold_factors['error'] = str(e)
        threshold_factors['final_threshold'] = adaptive_threshold
    
    threshold_time = time.time() - threshold_start
    print(f"Threshold calculation time: {threshold_time:.3f} seconds")
    print(f"Threshold factors: {threshold_factors}")
    
    # Calculate total processing time
    total_time = time.time() - start_total
    print(f"Total recognition time: {total_time:.3f} seconds")
    
    # Check if similarity exceeds adaptive threshold
    if best_similarity >= adaptive_threshold:
        try:
            # Map back to original index with better error handling
            match_index = mapped_ids.index(best_match_id)
            if match_index < len(valid_indices):
                original_idx = valid_indices[match_index]
                return original_idx, best_similarity, "Face recognized"
            else:
                print(f"Index mapping error: match_index={match_index}, valid_indices length={len(valid_indices)}")
                return None, best_similarity, "Error in student mapping"
        except Exception as e:
            print(f"Error mapping student index: {str(e)}")
            return None, best_similarity, f"Error in student mapping: {str(e)}"
    else:
        # Even with lower threshold, no match found
        return None, best_similarity, "No match found with sufficient confidence"