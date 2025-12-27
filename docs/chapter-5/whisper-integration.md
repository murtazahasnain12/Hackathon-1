# Whisper Integration

## Introduction to Whisper for Robotics

OpenAI's Whisper represents a breakthrough in automatic speech recognition (ASR), offering multilingual capabilities and robust performance across diverse acoustic conditions. For robotics applications, Whisper provides the ability to accurately transcribe human speech into text, which can then be processed by language models to understand commands and requests.

Whisper's architecture as a large-scale multilingual speech encoder makes it particularly valuable for robotics:
- **Multilingual Support**: Recognize commands in multiple languages
- **Robustness**: Perform well in noisy environments typical of robotic applications
- **Real-time Capability**: Process speech streams for interactive applications
- **Open Source**: Available for deployment in robotic systems

## Whisper Architecture Overview

### Transformer-Based Architecture

Whisper uses a transformer encoder-decoder architecture similar to GPT, but trained specifically on speech data:

#### Encoder
- **Mel-scale Spectrogram Input**: Processes audio as a spectrogram
- **Learned Audio Priors**: Encodes audio into high-level representations
- **Cross-Attention**: Enables decoder to attend to audio features

#### Decoder
- **Text Generation**: Produces text tokens autoregressively
- **Language Identification**: Identifies the language of the input
- **Timestamp Prediction**: Aligns text with audio timestamps

### Training Approach

#### Multitask Training
Whisper is trained jointly on multiple tasks:
- **Transcription**: Convert speech to text
- **Translation**: Translate speech to English text
- **Language Identification**: Identify input language
- **Timestamp Prediction**: Align text with audio

#### Data Efficiency
The model leverages large-scale audio-text datasets:
- **680,000 hours** of multilingual training data
- **Diverse acoustic conditions**: Noisy environments, accents, languages
- **Robust representations**: Generalizes to new acoustic conditions

## Whisper for Robotic Applications

### Audio Preprocessing

#### Microphone Array Processing
Robotic systems often use microphone arrays for better audio capture:

```python
import numpy as np
import scipy.signal as signal

def preprocess_robot_audio(audio_signal, sample_rate=16000):
    """
    Preprocess audio for Whisper ASR in robotic applications
    """
    # Apply beamforming to focus on speaker direction
    # (Assuming multi-channel audio)
    if audio_signal.ndim > 1:
        # Simple delay-and-sum beamforming
        processed_signal = beamform_audio(audio_signal, sample_rate)
    else:
        processed_signal = audio_signal

    # Denoise using spectral subtraction
    denoised_signal = spectral_subtract(processed_signal, sample_rate)

    # Normalize audio levels
    normalized_signal = normalize_audio(denoised_signal)

    return normalized_signal

def beamform_audio(audio_channels, sample_rate):
    """
    Apply delay-and-sum beamforming to focus on speaker
    """
    # Calculate delays for each microphone based on geometry
    delays = calculate_delays(sample_rate)  # Implementation depends on mic geometry

    # Apply delays and sum channels
    beamformed = np.zeros_like(audio_channels[0])
    for i, channel in enumerate(audio_channels):
        delayed = np.roll(channel, int(delays[i] * sample_rate))
        beamformed += delayed

    return beamformed
```

#### Noise Reduction
Robot environments often contain noise from motors, fans, and other sources:

```python
from scipy import signal
import webrtcvad

def adaptive_noise_reduction(audio, sample_rate):
    """
    Apply adaptive noise reduction for robotic environments
    """
    # Use Voice Activity Detection to identify speech segments
    vad = webrtcvad.Vad()

    # Apply spectral subtraction
    enhanced_audio = spectral_subtract(audio, sample_rate)

    # Apply Wiener filtering
    filtered_audio = wiener_filter(enhanced_audio, sample_rate)

    return filtered_audio

def spectral_subtract(noisy_signal, sample_rate):
    """
    Apply spectral subtraction for noise reduction
    """
    # Calculate FFT of signal
    fft = np.fft.fft(noisy_signal)

    # Estimate noise spectrum during non-speech segments
    noise_spectrum = estimate_noise_spectrum(noisy_signal)

    # Subtract noise spectrum
    enhanced_spectrum = np.maximum(
        np.abs(fft) - noise_spectrum, 0
    ) * np.exp(1j * np.angle(fft))

    # Inverse FFT
    enhanced_signal = np.real(np.fft.ifft(enhanced_spectrum))

    return enhanced_signal
```

### Real-Time Whisper Integration

#### Streaming ASR
For interactive robotic applications, streaming ASR is essential:

```python
import torch
import whisper
from transformers import pipeline
import threading
import queue
import time

class StreamingWhisperASR:
    def __init__(self, model_name="small", language="en"):
        # Load Whisper model
        self.model = whisper.load_model(model_name)
        self.language = language

        # Audio buffer for streaming
        self.audio_buffer = []
        self.buffer_size = 8000  # 0.5 seconds at 16kHz
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Result buffer
        self.transcription_history = []

    def add_audio_chunk(self, audio_chunk):
        """
        Add audio chunk to processing queue
        """
        self.processing_queue.put(audio_chunk)

    def process_audio_stream(self):
        """
        Continuously process audio chunks and transcribe
        """
        while True:
            try:
                audio_chunk = self.processing_queue.get(timeout=1.0)

                # Add to buffer
                self.audio_buffer.extend(audio_chunk)

                # Process if buffer is large enough
                if len(self.audio_buffer) >= self.buffer_size:
                    # Process audio (in real implementation, this would be more complex)
                    transcription = self.transcribe_buffer()

                    if transcription and transcription.strip():
                        self.results_queue.put(transcription)

                    # Clear old buffer (keep some overlap for continuity)
                    self.audio_buffer = self.audio_buffer[self.buffer_size//2:]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue

    def transcribe_buffer(self):
        """
        Transcribe current audio buffer
        """
        if len(self.audio_buffer) < 1600:  # Minimum for Whisper
            return ""

        # Convert to numpy array
        audio_array = np.array(self.audio_buffer, dtype=np.float32)

        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))

        # Transcribe using Whisper
        result = self.model.transcribe(
            audio_array,
            language=self.language,
            fp16=torch.cuda.is_available()
        )

        return result["text"]

    def get_latest_transcription(self):
        """
        Get latest transcription result
        """
        transcriptions = []
        try:
            while True:
                transcription = self.results_queue.get_nowait()
                transcriptions.append(transcription)
                self.transcription_history.append(transcription)
        except queue.Empty:
            pass

        # Return the most recent transcription
        return transcriptions[-1] if transcriptions else ""
```

#### Wake Word Detection
Combine wake word detection with Whisper for efficient processing:

```python
import speech_recognition as sr
import pvporcupine
import pyaudio

class WakeWordWhisper:
    def __init__(self, wake_words=["robot", "hey robot"]):
        # Initialize Porcupine wake word detector
        self.porcupine = pvporcupine.create(keywords=wake_words)

        # Initialize Whisper ASR
        self.whisper_asr = StreamingWhisperASR()

        # Audio stream parameters
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        self.is_listening = False
        self.listening_timeout = 5.0  # seconds
        self.listen_start_time = 0

    def start_listening_loop(self):
        """
        Main listening loop for wake word and command recognition
        """
        while True:
            # Read audio frames
            pcm = self.stream.read(self.porcupine.frame_length)
            pcm = np.frombuffer(pcm, dtype=np.int16)

            # Check for wake word
            keyword_index = self.porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")
                self.activate_listening()

            # Process audio if actively listening
            if self.is_listening:
                # Add to Whisper processing
                self.whisper_asr.add_audio_chunk(pcm.astype(np.float32) / 32768.0)

                # Check for timeout
                if time.time() - self.listen_start_time > self.listening_timeout:
                    self.deactivate_listening()

            # Check for commands if actively listening
            transcription = self.whisper_asr.get_latest_transcription()
            if transcription:
                self.process_command(transcription)
                self.whisper_asr.transcription_history = []  # Clear after processing

    def activate_listening(self):
        """
        Activate command listening mode
        """
        self.is_listening = True
        self.listen_start_time = time.time()
        print("Listening for commands...")

    def deactivate_listening(self):
        """
        Deactivate command listening mode
        """
        self.is_listening = False
        print("Command listening deactivated")

    def process_command(self, command):
        """
        Process recognized command
        """
        print(f"Recognized command: {command}")

        # Send command to language understanding module
        # This would typically be a ROS service call or topic publish
        self.send_command_to_understanding(command)
```

## Whisper Integration with ROS

### Whisper ROS Node

Implement Whisper as a ROS node for robotics applications:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Detection2DArray
import whisper
import torch
import numpy as np
import threading
import queue


class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_asr_node')

        # Initialize Whisper model
        self.model = whisper.load_model("small.en")  # or another model
        self.is_processing = False
        self.audio_buffer = []

        # ROS publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )

        self.transcription_pub = self.create_publisher(
            String,
            '/whisper/transcription',
            10
        )

        self.activation_sub = self.create_subscription(
            Bool,
            '/voice_activation',
            self.activation_callback,
            10
        )

        self.is_active = False

        # Processing queue for threading
        self.process_queue = queue.Queue()
        self.process_thread = threading.Thread(target=self.process_audio_queue)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.get_logger().info('Whisper ASR Node initialized')

    def activation_callback(self, msg):
        """
        Handle voice activation messages
        """
        self.is_active = msg.data
        if self.is_active:
            self.get_logger().info('Voice recognition activated')
        else:
            self.get_logger().info('Voice recognition deactivated')

    def audio_callback(self, msg):
        """
        Process incoming audio data
        """
        if not self.is_active or self.is_processing:
            return

        # Convert audio data
        audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to processing queue
        self.process_queue.put(audio_data)

    def process_audio_queue(self):
        """
        Process audio in separate thread to avoid blocking
        """
        while rclpy.ok():
            try:
                audio_chunk = self.process_queue.get(timeout=1.0)

                if not self.is_active:
                    continue

                self.is_processing = True

                # Transcribe audio
                transcription = self.transcribe_audio(audio_chunk)

                if transcription and transcription.strip():
                    # Publish transcription
                    transcription_msg = String()
                    transcription_msg.data = transcription
                    self.transcription_pub.publish(transcription_msg)
                    self.get_logger().info(f'Transcribed: {transcription}')

                self.is_processing = False

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error processing audio: {e}')
                self.is_processing = False

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio data using Whisper
        """
        try:
            # Ensure audio is in correct format
            if len(audio_data) < 1600:  # Minimum for Whisper
                return ""

            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                language='en',
                fp16=torch.cuda.is_available()
            )

            return result["text"]

        except Exception as e:
            self.get_logger().error(f'Whisper transcription error: {e}')
            return ""

    def destroy_node(self):
        """
        Cleanup on node destruction
        """
        self.process_queue.queue.clear()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperNode()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Performance Optimization

#### Model Quantization
Optimize Whisper for edge deployment:

```python
import torch
import whisper

def quantize_whisper_model(model_name="small"):
    """
    Quantize Whisper model for efficient edge deployment
    """
    # Load model
    model = whisper.load_model(model_name)

    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d},
        dtype=torch.qint8
    )

    return quantized_model

def optimize_whisper_pipeline():
    """
    Optimize Whisper pipeline for robotic applications
    """
    # Use smaller model for real-time applications
    # Load with appropriate precision
    model = whisper.load_model("tiny.en", device="cuda" if torch.cuda.is_available() else "cpu")

    # Configure for streaming
    # Set appropriate processing chunk sizes
    return model
```

#### Memory Management
Efficient memory usage for robotic systems:

```python
class MemoryEfficientWhisper:
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.model = None
        self.load_model()

        # Memory management parameters
        self.max_buffer_size = 30 * 16000  # 30 seconds of audio at 16kHz
        self.audio_buffer = np.array([], dtype=np.float32)

    def load_model(self):
        """
        Load Whisper model with memory considerations
        """
        if torch.cuda.is_available():
            self.model = whisper.load_model(self.model_size, device="cuda")
        else:
            # For CPU-only systems, use smaller model
            if self.model_size in ["large", "medium"]:
                self.model = whisper.load_model("small", device="cpu")
            else:
                self.model = whisper.load_model(self.model_size, device="cpu")

    def process_audio_stream(self, audio_chunk):
        """
        Process audio stream with memory efficiency
        """
        # Add chunk to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Process when buffer is large enough
        if len(self.audio_buffer) >= 16000:  # 1 second at 16kHz
            # Transcribe
            result = self.model.transcribe(self.audio_buffer[:16000])

            # Remove processed audio
            self.audio_buffer = self.audio_buffer[16000:]

            return result["text"]

        return ""

    def clear_memory(self):
        """
        Clear audio buffer to free memory
        """
        self.audio_buffer = np.array([], dtype=np.float32)
```

## Whisper in VLA Pipelines

### Vision-Language-Audio Integration

Whisper enables the "A" in VLA (Vision-Language-Action) pipelines by providing audio input processing:

#### Audio Command Processing
```python
class VLAPipeline:
    def __init__(self):
        # Initialize Whisper for audio processing
        self.whisper = whisper.load_model("small.en")

        # Initialize vision system
        self.vision_system = VisionSystem()

        # Initialize language understanding
        self.language_model = LanguageModel()

        # Initialize action planning
        self.action_planner = ActionPlanner()

    def process_command(self, audio_input, visual_input):
        """
        Process VLA command: Vision + Language (from audio) + Action
        """
        # Step 1: Convert audio to text using Whisper
        text_command = self.whisper.transcribe(audio_input)["text"]

        # Step 2: Combine visual and linguistic information
        combined_input = self.combine_visual_language(visual_input, text_command)

        # Step 3: Plan appropriate action
        action_plan = self.plan_action(combined_input)

        # Step 4: Execute action
        self.execute_action(action_plan)

    def combine_visual_language(self, visual_features, text_command):
        """
        Combine visual and linguistic information
        """
        # Process text command
        command_embedding = self.language_model.encode(text_command)

        # Process visual information
        visual_features = self.vision_system.extract_features(visual_input)

        # Combine modalities (this would be a learned fusion mechanism)
        combined_features = torch.cat([visual_features, command_embedding], dim=-1)

        return combined_features
```

### Multimodal Command Understanding

Whisper enables complex multimodal command understanding:

#### Example: "Pick up the red cup near the window"
1. **Audio**: Whisper transcribes "Pick up the red cup near the window"
2. **Language**: NLP processes the command, identifying "red cup" as the target object and "near the window" as spatial context
3. **Vision**: Processes the scene to identify red cups and the window location
4. **Action**: Plans a trajectory to approach the identified object

```python
def multimodal_command_pipeline(audio_input, image_input):
    """
    Process multimodal command combining audio, vision, and action
    """
    # Step 1: Audio to text (Whisper)
    text = whisper_model.transcribe(audio_input)["text"]

    # Step 2: Language understanding
    command_structure = parse_command_structure(text)
    # Returns: {"action": "pick up", "object": "red cup", "spatial_context": "near window"}

    # Step 3: Visual object detection with spatial context
    detected_objects = vision_system.detect_objects(image_input)
    target_object = identify_target_object(detected_objects, command_structure)

    # Step 4: Action planning and execution
    action_plan = action_planner.create_plan(target_object, command_structure["action"])
    robot.execute(action_plan)
```

## Whisper Deployment on Edge Platforms

### Jetson Integration

Deploy Whisper on NVIDIA Jetson platforms for robotics:

```python
class JetsonWhisper:
    def __init__(self):
        # Check for Jetson platform
        self.is_jetson = self.check_jetson_platform()

        if self.is_jetson:
            # Optimize for Jetson
            self.optimize_for_jetson()

        # Load appropriate model
        self.model = self.load_optimized_model()

    def check_jetson_platform(self):
        """
        Check if running on Jetson platform
        """
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                return 'jetson' in model.lower()
        except:
            return False

    def optimize_for_jetson(self):
        """
        Apply Jetson-specific optimizations
        """
        # Use TensorRT for acceleration if available
        # Set appropriate model size based on Jetson variant
        pass

    def load_optimized_model(self):
        """
        Load model optimized for Jetson
        """
        if self.is_jetson:
            # Use smaller model or quantized model
            return whisper.load_model("tiny.en", device="cuda")
        else:
            return whisper.load_model("small.en", device="cuda" if torch.cuda.is_available() else "cpu")
```

### Resource-Constrained Deployment

Optimize Whisper for resource-constrained robotic systems:

```python
class EfficientWhisperDeployment:
    def __init__(self, max_memory_mb=1000):
        self.max_memory_mb = max_memory_mb
        self.model = self.load_memory_efficient_model()
        self.setup_processing_pipeline()

    def load_memory_efficient_model(self):
        """
        Load model considering memory constraints
        """
        if self.max_memory_mb < 500:
            # Use tiny model or quantized model
            return whisper.load_model("tiny.en")
        elif self.max_memory_mb < 1000:
            return whisper.load_model("small.en")
        else:
            return whisper.load_model("medium.en")

    def setup_processing_pipeline(self):
        """
        Setup pipeline with processing chunk sizes based on resources
        """
        # Determine optimal chunk sizes based on available memory
        self.chunk_size = self.determine_optimal_chunk_size()

    def determine_optimal_chunk_size(self):
        """
        Determine optimal processing chunk size based on resources
        """
        # Calculate based on available memory and model size
        # Return appropriate chunk size for real-time processing
        return 8000  # 0.5 seconds at 16kHz
```

## Integration with Language Models

### Whisper + LLM Integration

Combine Whisper with Large Language Models for command understanding:

```python
class WhisperLLMIntegration:
    def __init__(self):
        # Initialize Whisper for ASR
        self.whisper = whisper.load_model("small.en")

        # Initialize LLM for command understanding
        self.llm = self.initialize_llm()

    def process_voice_command(self, audio_input):
        """
        Process voice command through Whisper and LLM
        """
        # Step 1: Speech to text
        transcription = self.whisper.transcribe(audio_input)
        text_command = transcription["text"]

        # Step 2: Command understanding with LLM
        structured_command = self.understand_command(text_command)

        return structured_command

    def understand_command(self, text_command):
        """
        Use LLM to understand and structure the command
        """
        prompt = f"""
        Parse the following robot command into structured format:
        Command: "{text_command}"

        Return the structured command in JSON format:
        {{
            "action": "...",
            "target_object": "...",
            "spatial_constraints": "...",
            "context": "..."
        }}
        """

        # Call LLM to parse command
        response = self.llm.generate(prompt)

        # Parse JSON response
        try:
            import json
            structured_command = json.loads(response)
            return structured_command
        except:
            # Fallback: basic parsing
            return self.basic_parse(text_command)
```

## Evaluation and Performance Metrics

### ASR Performance in Robotics

Evaluate Whisper performance in robotic environments:

#### Word Error Rate (WER)
```python
def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate for ASR evaluation
    """
    import editdistance

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return wer
```

#### Robustness Metrics
- **Signal-to-Noise Ratio (SNR) Robustness**: ASR performance under varying noise levels
- **Acoustic Condition Robustness**: Performance in different acoustic environments
- **Speaker Adaptation**: Performance across different speakers

### Robotics-Specific Metrics

#### Command Understanding Accuracy
- **Intent Recognition**: Accuracy of understanding the command intent
- **Entity Recognition**: Accuracy of identifying objects and spatial relations
- **Action Mapping**: Accuracy of mapping commands to appropriate robot actions

#### Real-Time Performance
- **Latency**: Time from audio input to action execution
- **Throughput**: Commands processed per second
- **Resource Utilization**: CPU/GPU memory and compute usage

## Troubleshooting and Common Issues

### Audio Quality Issues
- **Low Signal-to-Noise Ratio**: Use beamforming or noise reduction
- **Clipping**: Normalize audio levels
- **Insufficient Volume**: Use audio gain appropriately

### Performance Issues
- **High Latency**: Use smaller models or optimize processing pipeline
- **Memory Issues**: Use quantized models or optimize buffer sizes
- **Accuracy Issues**: Fine-tune on domain-specific audio data

## Summary

Whisper integration provides powerful speech recognition capabilities for robotics applications, enabling natural voice-based interaction with humanoid robots. The combination of robust ASR with language understanding models creates the foundation for comprehensive voice command systems.

Key considerations for successful Whisper integration in robotics include:
- Audio preprocessing for robotic environments
- Real-time processing capabilities
- Integration with ROS messaging systems
- Performance optimization for edge platforms
- Multimodal fusion with vision and language systems

As robotics applications continue to evolve, the integration of advanced speech recognition like Whisper will enable increasingly natural and intuitive human-robot interaction, moving closer to truly conversational robotic assistants.

The next section will explore how these speech and language capabilities are connected to robotic action systems through LLM-ROS integration.

## Navigation Links

- **Previous**: [Vision-Language Fundamentals](./vision-language.md)
- **Next**: [LLM-ROS Integration](./llm-ros-actions.md)
- **Up**: [Chapter 5](./index.md)

## Next Steps

Continue learning about how speech and language capabilities are connected to robotic action systems through LLM-ROS integration.