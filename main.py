import io
import logging
import os
import threading
import time
import socket

import requests
from PIL import Image
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import PushMatrix, PopMatrix, Rotate
from kivy.lang import Builder
from kivy.logger import Logger
# Explicitly import KivyMD classes for 1.2.0
from kivy.properties import ObjectProperty, BooleanProperty, StringProperty, NumericProperty
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.utils import platform
from kivymd.app import MDApp
from plyer import tts, filechooser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from android.permissions import request_permissions, Permission
request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO, Permission.RECORD_AUDIO])
# Configure logging
logging.basicConfig(level=logging.INFO)
for module in ['comtypes', 'urllib3', 'PIL']:
    logging.getLogger(module).setLevel(logging.WARNING)

# Global TTS management
tts_engine = None
tts_lock = threading.Lock()
tts_initialized = False

# Configuration
TEST_MODE = (platform == 'win')  # True on Windows, False on Android

# SERVER CONFIGURATION - IMPORTANT CHANGE HERE
# Default server URL that will be updated dynamically
SERVER_URL = "http://127.0.0.1:8000/process-image" if TEST_MODE else None
SERVER_BASE = "http://127.0.0.1:8000" if TEST_MODE else None
SERVER_DISCOVERY_TIMEOUT = 2  # Seconds to wait for server discovery
SERVER_DISCOVERY_PORTS = [8000]  # Ports to try during discovery
DISCOVERY_IPS = []  # Will be populated during discovery

TIMEOUT = 120
CAPTURES_DIR = "captures"

# Initialize directories upfront
os.makedirs(CAPTURES_DIR, exist_ok=True)

# FIX 1: Disable Windows-specific providers that cause crashes on exit
if TEST_MODE:
    from kivy.config import Config

    Config.set('input', 'wm_pen', '0')
    Config.set('input', 'wm_touch', '0')

# Set camera backend preference
from kivy.config import Config

Config.set('kivy', 'camera', 'opencv')  # Prefer OpenCV backend


def init_tts():
    """Initialize text-to-speech engine based on platform"""
    global tts_engine, tts_initialized
    if not tts_initialized:
        try:
            if platform == 'android':
                tts_initialized = True
                Logger.info("TTS: Android TTS initialized")
            elif TEST_MODE:
                import pyttsx3
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                tts_initialized = True
                Logger.info("TTS: Windows TTS initialized")
        except Exception as e:
            Logger.error(f"TTS init failed: {e}")


def shutdown_tts():
    """Safely shut down TTS engine"""
    global tts_engine, tts_initialized
    with tts_lock:
        if tts_engine and hasattr(tts_engine, 'stop'):
            try:
                tts_engine.stop()
            except Exception as e:
                Logger.error(f"TTS shutdown error: {e}")
            tts_engine = None
            tts_initialized = False


# Set window size for desktop testing
if TEST_MODE:
    Window.size = (360, 640)


# NEW: Server discovery function
def discover_server():
    """
    Discover the image processing server on the network
    Returns server base URL if found, None otherwise
    """
    global SERVER_URL, SERVER_BASE, DISCOVERY_IPS

    if TEST_MODE:
        Logger.info("Test mode: Using localhost server")
        return "http://127.0.0.1:8000"

    Logger.info("Starting server discovery...")

    # Get all possible IP addresses in the local network
    DISCOVERY_IPS = []

    # First try to get our own IP to determine subnet
    try:
        # Create a socket to determine our IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external server (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        our_ip = s.getsockname()[0]
        s.close()

        # Get the subnet part (first 3 octets)
        subnet = '.'.join(our_ip.split('.')[:3])
        Logger.info(f"Our IP: {our_ip}, Subnet: {subnet}")

        # Add our own IP first
        DISCOVERY_IPS.append(our_ip)

        # Add specific IPs that are likely to be the server
        for last_octet in range(1, 256):
            ip = f"{subnet}.{last_octet}"
            if ip != our_ip:
                DISCOVERY_IPS.append(ip)

    except Exception as e:
        Logger.error(f"Error getting subnet: {e}")
        # Fallback to some common IPs
        DISCOVERY_IPS = ["192.168.1.5", "192.168.0.5", "10.0.0.5", "10.0.2.2"]

    # Always include localhost and 10.0.2.2 (Android emulator host)
    if "127.0.0.1" not in DISCOVERY_IPS:
        DISCOVERY_IPS.append("127.0.0.1")
    if "10.0.2.2" not in DISCOVERY_IPS:
        DISCOVERY_IPS.append("10.0.2.2")

    Logger.info(f"Will try these IPs: {DISCOVERY_IPS}")

    # Try each IP and port combination
    session = requests.Session()
    session.mount('http://', HTTPAdapter(max_retries=0))  # No retries for faster scanning

    for ip in DISCOVERY_IPS:
        for port in SERVER_DISCOVERY_PORTS:
            try:
                test_url = f"http://{ip}:{port}/health"
                Logger.info(f"Trying {test_url}")

                response = session.get(test_url, timeout=SERVER_DISCOVERY_TIMEOUT)
                if response.status_code == 200:
                    server_base = f"http://{ip}:{port}"
                    Logger.info(f"Server found at: {server_base}")
                    return server_base
            except Exception as e:
                Logger.debug(f"No server at {ip}:{port}: {str(e)}")
                continue

    Logger.warning("No server found on network")
    return None


class MainScreen(Screen):
    """Main application screen with voice command listener"""
    glasses_icon = ObjectProperty(None)
    status_text = StringProperty("Ready - say 'take picture' or select an image")
    processing = BooleanProperty(False)
    animation_speed = NumericProperty(1.5)
    _anim = None  # Track animation to stop later

    def on_kv_post(self, *args):
        """Initialize animations after KV file is loaded"""
        self.animate_icon()
        Clock.schedule_once(self.server_discovery, 1)  # Try server discovery early
        Clock.schedule_once(self.initial_prompt, 4)

    def server_discovery(self, dt):
        """Start server discovery in the background"""
        threading.Thread(target=self._discover_server_thread, daemon=True).start()

    def _discover_server_thread(self):
        """Background thread for server discovery"""
        global SERVER_URL, SERVER_BASE

        try:
            result = discover_server()
            if result:
                SERVER_BASE = result
                SERVER_URL = f"{SERVER_BASE}/process-image"
                self.update_status(f"Server found at {SERVER_BASE}")
            else:
                self.update_status("Server not found! Check connection.")
                self.speak_feedback("Server not found. Please check connection settings.")
        except Exception as e:
            Logger.error(f"Server discovery error: {e}")
            self.update_status("Server discovery error")

    def on_enter(self):
        """Actions when entering this screen"""
        self.processing = False
        self.status_text = "Ready - say 'take picture' or select an image"
        # Try server discovery again when returning to main screen
        if not SERVER_BASE:
            Clock.schedule_once(self.server_discovery, 0.5)
        Clock.schedule_once(self.auto_listen, 2)

    def on_leave(self):
        """Clean up when leaving this screen"""
        if self._anim:
            self._anim.cancel(self.glasses_icon)

    def animate_icon(self):
        """Create floating animation for glasses icon"""
        if self.glasses_icon:
            self._anim = (Animation(pos_hint={"center_y": 0.52}, duration=self.animation_speed) +
                          Animation(pos_hint={"center_y": 0.48}, duration=self.animation_speed))
            self._anim.repeat = True
            self._anim.start(self.glasses_icon)

    def initial_prompt(self, dt):
        """Initial voice prompt to user"""
        self.status_text = "Say 'Take picture' to get started"
        self.speak_feedback("Say Take picture to get started")
        Clock.schedule_once(self.auto_listen, 1)

    def auto_listen(self, dt):
        """Automatically start listening if not processing"""
        if not self.processing:
            self.listen_for_command()

    def go_to_camera(self):
        """Navigate to camera screen"""
        self.manager.transition = FadeTransition(duration=0.5)
        self.manager.current = 'camera'

    def go_to_loading(self, image_path=None):
        """Navigate to loading screen with optional image path"""
        loading_screen = self.manager.get_screen('loading')
        if image_path:
            loading_screen.image_to_process = image_path
        self.manager.transition = FadeTransition(duration=0.5)
        self.manager.current = 'loading'

    def listen_for_command(self):
        """Start listening for voice commands"""
        if self.processing:
            return
        self.processing = True
        threading.Thread(target=self._listen_thread, daemon=True).start()

    def _listen_thread(self):
        """Background thread for speech recognition"""
        try:
            # Lazy import speech_recognition to avoid startup errors
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            try:
                with sr.Microphone() as source:
                    self.update_status("Listening...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                command = recognizer.recognize_google(audio).lower()
                self.update_status(f"Heard: {command}")

                if "take picture" in command or "photo" in command:
                    Clock.schedule_once(lambda dt: self.go_to_camera())
                else:
                    self.speak_feedback("Please say 'take picture' or use the select button")
                    Clock.schedule_once(self.auto_listen, 2)
            except sr.UnknownValueError:
                self.update_status("Couldn't understand audio")
                self.speak_feedback("I didn't hear that clearly. Please try again.")
                Clock.schedule_once(self.auto_listen, 2)
            except sr.WaitTimeoutError:
                self.update_status("Listening timed out")
                self.speak_feedback("I didn't hear anything. Please try again.")
                Clock.schedule_once(self.auto_listen, 2)
            except Exception as e:
                self.update_status(f"Error: {str(e)}")
                Logger.error(f"Voice command error: {str(e)}")
                Clock.schedule_once(self.auto_listen, 2)
        except ImportError:
            self.update_status("Speech recognition not available")
            Logger.error("Speech recognition module not available")
        finally:
            self.processing = False

    def choose_image(self):
        """Open file chooser to select image"""
        if self.processing:
            return
        self.processing = True
        self.update_status("Opening file chooser...")
        try:
            if platform == 'android':
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.READ_EXTERNAL_STORAGE])

            filechooser.open_file(
                on_selection=self.handle_selection,
                filters=[("Images", "*.jpg", "*.jpeg", "*.png", "*.bmp")]
            )
        except Exception as e:
            self.update_status(f"File chooser error: {str(e)}")
            Logger.error(f"FileChooser: {str(e)}")
            self.processing = False

    def handle_selection(self, selection):
        """Handle selected file from chooser"""
        if not selection:
            self.processing = False
            return
        try:
            filepath = selection[0]
            Logger.info(f"Selected file: {filepath}")

            if not filepath.lower().endswith(('.jpg', '.jpeg')):
                try:
                    self.update_status("Converting image format...")
                    jpg_path = os.path.join(CAPTURES_DIR, 'converted_image.jpg')
                    with Image.open(filepath) as img:
                        img.convert('RGB').save(jpg_path, 'JPEG', quality=85)
                    filepath = jpg_path
                    Logger.info(f"Image converted to JPEG: {filepath}")
                except IOError as e:
                    Logger.error(f"Image conversion error: {str(e)}")
                    raise

            self.go_to_loading(filepath)
        except Exception as e:
            Logger.error(f"Error handling selection: {str(e)}")
            self.update_status("Error selecting file")
            self.processing = False

    def update_status(self, text):
        """Update status text safely from background thread"""
        Clock.schedule_once(lambda dt: setattr(self, 'status_text', text))

    def speak_feedback(self, text):
        """Speak text feedback to user"""
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        """Background thread for TTS"""
        try:
            if platform == 'android':
                tts.speak(text)
            elif TEST_MODE:
                init_tts()
                global tts_engine
                if tts_initialized and tts_engine is not None:
                    with tts_lock:
                        try:
                            if hasattr(tts_engine, 'stop') and callable(tts_engine.stop):
                                tts_engine.stop()
                            tts_engine.say(text)
                            tts_engine.runAndWait()
                        except Exception as e:
                            Logger.error(f"TTS speaking error: {str(e)}")
                            tts_engine = None
                            init_tts()
        except Exception as e:
            Logger.error(f"TTS error: {str(e)}")


class CameraScreen(Screen):
    """Screen for camera capture"""
    camera = ObjectProperty(None)
    status_text = StringProperty("Preparing to capture image...")
    camera_initialized = BooleanProperty(False)
    _camera_index = 0  # Track which camera we're using

    def on_enter(self):
        """Setup when entering screen"""
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA])
        Clock.schedule_once(self.init_camera, 0.5)

    def on_leave(self):
        """Cleanup when leaving screen"""
        self.stop_camera()

    def stop_camera(self):
        """Safely stop the camera"""
        if self.camera and self.camera_initialized:
            try:
                self.camera.play = False
                self.camera_initialized = False
            except Exception as e:
                Logger.error(f"Error stopping camera: {e}")

    def init_camera(self, dt, retry_count=0):
        """Initialize camera with fallbacks"""
        try:
            if self.camera:
                # Safety cleanup first
                try:
                    if hasattr(self.camera, 'play'):
                        self.camera.play = False
                except Exception:
                    pass

                # Try different resolutions if default fails
                resolutions = [(640, 480), (800, 600), (1280, 720)]

                # Try to find a working camera and resolution
                for res in resolutions:
                    try:
                        # Set index for Windows to try different cameras
                        if TEST_MODE and retry_count > 0:
                            self._camera_index = (self._camera_index + 1) % 3
                            self.camera.index = self._camera_index
                            Logger.info(f"Trying camera index: {self._camera_index}")

                        self.camera.resolution = res
                        self.camera.play = True

                        # Wait a bit and check if camera is actually working
                        Clock.schedule_once(lambda dt: self.check_camera_working(res), 1)
                        return
                    except Exception as e:
                        Logger.warning(f"Failed at resolution {res}: {str(e)}")
                        continue

                # If all resolutions fail
                raise Exception("Could not initialize camera at any resolution")

        except Exception as e:
            Logger.error(f"Camera init error: {str(e)}")
            if retry_count < 2:
                self.status_text = f"Retrying camera ({retry_count + 1}/2)..."
                Clock.schedule_once(lambda dt: self.init_camera(dt, retry_count + 1), 1)
            else:
                self.status_text = "Camera unavailable"
                Logger.error("Camera initialization failed")
                Clock.schedule_once(lambda dt: (
                    self.manager.get_screen('main').speak_feedback("Camera not available"),
                    setattr(self.manager, 'current', 'main')
                ), 2)

    def check_camera_working(self, resolution):
        """Verify camera is actually working"""
        if self.camera and hasattr(self.camera, 'texture') and self.camera.texture:
            self.camera_initialized = True
            self.status_text = f"Camera ready at {resolution[0]}x{resolution[1]}"
            Clock.schedule_once(self.auto_capture, 1)
        else:
            Logger.warning("Camera texture not available, retrying...")
            self.camera_initialized = False
            self.stop_camera()
            Clock.schedule_once(lambda dt: self.init_camera(dt, 1), 0.5)

    def auto_capture(self, dt):
        """Auto-capture after camera is ready"""
        if self.camera_initialized:
            self.capture_image()
        else:
            self.status_text = "Camera not ready - trying again"
            Clock.schedule_once(self.auto_capture, 1)

    def capture_image(self):
        """Capture image with fallback strategies"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            jpg_path = os.path.join(CAPTURES_DIR, f"photo_{timestamp}.jpg")

            # Check if camera texture exists before trying to capture
            if not self.camera or not hasattr(self.camera, 'texture') or not self.camera.texture:
                raise ValueError("Camera texture not available")

            # FIX 2: Fixed CoreImage capture method
            try:
                from kivy.core.image import Image as CoreImage
                from kivy.graphics.texture import Texture

                # Get texture from camera
                texture = self.camera.texture
                if not isinstance(texture, Texture):
                    raise TypeError("Invalid texture format")

                # Convert texture to PIL Image
                pixels = texture.pixels
                size = texture.size
                pil_image = Image.frombytes('RGBA', size, pixels)

                # Convert to RGB and save as JPEG
                pil_image.convert('RGB').save(jpg_path, 'JPEG', quality=85)
                Logger.info(f"Image captured successfully: {jpg_path}")
                self.manager.get_screen('loading').image_to_process = jpg_path
                self.manager.current = 'loading'
                return
            except Exception as e:
                Logger.warning(f"CoreImage capture method failed: {str(e)}")

            # Fallback to PNG then convert
            temp_path = os.path.join(CAPTURES_DIR, f"temp_{timestamp}.png")
            try:
                self.camera.export_to_png(temp_path)
                with Image.open(temp_path) as img:
                    img.convert("RGB").save(jpg_path, "JPEG", quality=85)
                os.remove(temp_path)
                self.manager.get_screen('loading').image_to_process = jpg_path
                self.manager.current = 'loading'
                return
            except Exception as e:
                Logger.error(f"PNG capture failed: {str(e)}")

            # Last resort: create blank image
            Logger.warning("Creating blank image as fallback")
            img = Image.new('RGB', (640, 480), color='gray')
            img.save(jpg_path)
            self.manager.get_screen('loading').image_to_process = jpg_path
            self.manager.current = 'loading'

        except Exception as e:
            self.status_text = "Capture failed"
            Logger.error(f"Final capture error: {str(e)}")
            self.manager.current = 'main'


class LoadingScreen(Screen):
    """Screen for loading and processing images"""
    image_to_process = StringProperty(None)
    status_text = StringProperty("Processing...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = self._create_session()
        self.wheel_animation = None

    def _create_session(self):
        """Create a requests session with retry capabilities"""
        session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def on_enter(self):
        """Setup when entering screen"""
        # Start wheel rotation animation
        self.start_wheel_animation()

        if self.image_to_process:
            Clock.schedule_once(lambda dt: self.process_image(self.image_to_process), 1)

    def on_leave(self):
        """Stop animations when leaving the screen"""
        self.stop_wheel_animation()

    def start_wheel_animation(self):
        """Start rotating the wheel icon"""
        try:
            wheel_image = self.ids.wheel_image
            if wheel_image:
                # Create rotation animation
                self.wheel_animation = Animation(angle=360, duration=2)
                self.wheel_animation.repeat = True

                # Start animation on the wheel image
                if hasattr(wheel_image, 'angle'):
                    wheel_image.angle = 0
                    self.wheel_animation.start(wheel_image)
                else:
                    # If angle property doesn't exist, create it dynamically
                    wheel_image.angle = 0

                    # Add rotation transform to the wheel
                    with wheel_image.canvas.before:
                        PushMatrix()
                        rot = Rotate(angle=0, origin=wheel_image.center, axis=(0, 0, 1))

                        # Bind rotation angle to wheel's angle property
                        def update_rot_angle(instance, value):
                            rot.angle = value

                        wheel_image.bind(angle=update_rot_angle)

                    with wheel_image.canvas.after:
                        PopMatrix()

                    self.wheel_animation.start(wheel_image)
        except Exception as e:
            Logger.error(f"Wheel animation error: {str(e)}")

    def stop_wheel_animation(self):
        """Stop wheel rotation animation"""
        try:
            if self.wheel_animation and hasattr(self, 'ids') and hasattr(self.ids, 'wheel_image'):
                self.wheel_animation.cancel(self.ids.wheel_image)
        except Exception as e:
            Logger.error(f"Stop animation error: {str(e)}")

    def process_image(self, image_path):
        """Start a thread to process the image"""
        threading.Thread(target=self._process_thread, args=(image_path,), daemon=True).start()

    def _process_thread(self, image_path):
        """Process image in a background thread"""
        try:
            global SERVER_URL, SERVER_BASE

            # Check if we have a valid server URL
            if not SERVER_BASE or not SERVER_URL:
                # Try to discover server again
                self.update_status("Finding server...")
                server_base = discover_server()
                if server_base:
                    SERVER_BASE = server_base
                    SERVER_URL = f"{SERVER_BASE}/process-image"
                    self.update_status(f"Connected to server at {SERVER_BASE}")
                else:
                    self.update_status("Server not found")
                    self.manager.get_screen('main').speak_feedback(
                        "Server not found. Please check your network connection.")
                    Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 5)
                    return

            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            self.update_status("Sending image to server...")
            Logger.info(f"Processing image: {image_path}")

            with Image.open(image_path) as img:
                # Convert image to in-memory buffer
                buffer = io.BytesIO()
                img.convert('RGB').save(buffer, format="JPEG", quality=85)
                buffer.seek(0)  # Reset buffer position to start
                file_size = buffer.getbuffer().nbytes
                Logger.info(f"Image size: {file_size} bytes")

                # Test mode: check server connection before sending
                if TEST_MODE:
                    try:
                        # Ping server to check if it's running
                        self.session.get(SERVER_URL.rsplit('/', 1)[0] + "/health", timeout=5)
                    except requests.exceptions.ConnectionError:
                        self.status_text = "Server not running - using mock response"
                        Logger.warning("Server not running, using mock data")

                        # Mock successful response for testing
                        self.status_text = "Done. Speaking result..."
                        mock_description = "This is a test response. Server connection was not available."
                        self.manager.get_screen('main').speak_feedback(mock_description)
                        Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 5)
                        return

                # Send image to server
                response = self.session.post(
                    SERVER_URL,
                    files={'file': (os.path.basename(image_path), buffer, 'image/jpeg')},
                    timeout=TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    Logger.info(f"Server response: {result}")
                    if 'description' in result:
                        self.update_status("Done. Speaking result...")
                        description = result['description']
                        self.manager.get_screen('main').speak_feedback(description)
                    else:
                        self.update_status("No description found")
                else:
                    self.update_status(f"Server error: {response.status_code}")
                    Logger.error(f"Server error {response.status_code}: {response.text}")
                    self.manager.get_screen('main').speak_feedback("Server error. Please try again.")
        except FileNotFoundError as e:
            self.update_status("Image file not found")
            Logger.error(f"File error: {str(e)}")
            self.manager.get_screen('main').speak_feedback("Image file not found. Please try again.")
        except requests.exceptions.ConnectionError:
            self.update_status("Server connection error")
            Logger.error("Could not connect to server")

            # Try to rediscover server
            server_base = discover_server()
            if server_base:
                SERVER_BASE = server_base
                SERVER_URL = f"{SERVER_BASE}/process-image"
                self.update_status(f"Found new server at {SERVER_BASE}")
                self.manager.get_screen('main').speak_feedback(
                    "Server connection restored. Please try again.")
            else:
                self.manager.get_screen('main').speak_feedback(
                    "Server connection error. Please check your internet connection.")
        except Exception as e:
            self.update_status("Failed to process image")
            Logger.error(f"Processing error: {str(e)}")
            self.manager.get_screen('main').speak_feedback("Failed to process the image. Please try again.")

        Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 5)

    def update_status(self, text):
        """Update status text safely from background thread"""
        Clock.schedule_once(lambda dt: setattr(self, 'status_text', text))


class MainApp(MDApp):
    """Main application class"""

    def build(self):
        """Build the application UI"""
        try:
            kv_path = os.path.join(os.path.dirname(__file__), "main.kv")
            if os.path.exists(kv_path):
                Builder.load_file(kv_path)
            else:
                # Fall back to inline KV string if file not found
                Logger.warning("main.kv file not found at: " + kv_path)
                return Label(text="KV file missing - please ensure main.kv is in the same directory")
        except Exception as e:
            Logger.error(f"KV load error: {str(e)}")
            return Label(text="UI Loading Error - Basic Mode")

        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MainScreen(name="main"))
        sm.add_widget(CameraScreen(name="camera"))
        sm.add_widget(LoadingScreen(name="loading"))
        return sm

    def on_start(self):
        """Actions when app starts"""
        # Initialize TTS if in test mode
        if TEST_MODE:
            init_tts()

        # Start server discovery
        if not TEST_MODE:
            # Try to discover server in background
            threading.Thread(
                target=discover_server,
                daemon=True
            ).start()

    def on_stop(self):
        """Clean up when app is closed"""
        try:
            # Clean up camera
            camera_screen = self.root.get_screen('camera')
            if camera_screen and camera_screen.camera:
                camera_screen.camera.play = False

            # Shut down TTS
            if TEST_MODE:
                shutdown_tts()
        except Exception as e:
            Logger.error(f"Cleanup error: {str(e)}")


# F
# FIX 5: Safer error handling for fatal errors
def safe_run_app():
    """Run the app with proper error handling"""
    try:
        app = MainApp()
        app.run()
    except Exception as e:
        # Log and show any critical errors
        error_msg = f"Fatal error: {str(e)}"
        Logger.critical(error_msg)

        # If on desktop, show error in a window using minimal dependencies
        if platform == 'win':
            try:
                # Very minimal error window
                import tkinter as tk
                root = tk.Tk()
                root.title("EyeSight Assistant - Error")
                tk.Label(root, text=error_msg, fg="red", padx=20, pady=20).pack()
                tk.Button(root, text="OK", command=root.destroy).pack(pady=10)
                root.mainloop()
            except Exception:
                # If even tkinter fails, just print to console
                print(f"CRITICAL ERROR: {error_msg}")


if __name__ == '__main__':
    safe_run_app()
