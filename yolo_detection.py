import os
import sys
import cv2
import time
import queue
import base64
import requests
import multiprocessing
from ultralytics import YOLO
from logger import get_logger, disable_console_logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from helper import *
from constants import *
from popup_window import *
from helper import get_config_value

disable_console_logging()
logger = get_logger(__name__)
image_queue = queue.Queue()
INPUT_EXTENSIONS = (".png", ".jpg", ".jpeg")
CUSTOMER_ID = get_config_value('CustomerID')
SUB_PROVISION_ID = get_config_value('SubProvisionID')
VERSIONID = get_config_value('VERSIONID')
SUBPROCESS_NAME = get_config_value('SUBPROCESS_NAME')
SERVER_TYPE = get_config_value('ServerType')
SERVER_TYPE = SERVER_TYPE if SERVER_TYPE else 1
# YOLO settings
MODEL_PATH = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
YOLO_MODEL_PATH = os.path.join(MODEL_PATH, f"{CUSTOMER_ID}_{SUB_PROVISION_ID}_{VERSIONID}.pt")
YOLO_CONFIG = get_config_value('YoloConfig')

if getattr(sys, 'frozen', False):
    sys.path = [p for p in sys.path if '_MEI' in p or p.endswith('.zip')]

def backup_image(image_path: str):
    """
    Send image as bits to backup server through API.
    """

    try:
        user_ntid = getpass.getuser()
        user_ntid = user_ntid.replace('.', '_')
        # convert image to bytes
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            encoded_bytes = base64.b64encode(image_data)
            encoded_string = encoded_bytes.decode('utf-8')

        image_name = image_path.split(".png")[0].split('\\')[-1]
        filename = f"{CUSTOMER_ID}_{SUB_PROVISION_ID}_{SUBPROCESS_NAME}_{user_ntid}_{image_name}.png"

        api_url = get_config_value("IMAGE_BACKUP_URL")
        headers = {'Content-Type': 'application/json'}
        context = {
            "filename": filename,
            "bitcode": encoded_string}
        response = requests.post(api_url, json.dumps(context), headers=headers)
        if response.status_code == 200:
            logger.info(f"Successfully backed up image: {filename}")
        else:
            logger.error(f"Failed to back up image. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error during image backup: {e}")

def yolo_prediction(image_queue: multiprocessing.Queue):
    start_time = time.time()
    from logger import get_logger   # re-import inside process
    logger = get_logger("worker")   # new logger instance

    model = YOLO(YOLO_MODEL_PATH).to("cpu")
    while True:
        image_path = image_queue.get()
        if image_path is None:  # exit signal
            break  # do NOT process further

        if not os.path.isfile(image_path):
            logger.error(f"File does not exist: {image_path}")
            continue

        try:
            logger.info(f"[+] Processing: {os.path.basename(image_path)}")

            for _ in range(5):
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    break
                time.sleep(2)

            time.sleep(3)
            start_time = time.time()
            img = cv2.imread(image_path)
            image_name = image_path.split(".png")[0].split('\\')[-1]

            # if int(SERVER_TYPE) == 3:
            #     logger.info(f"[+] Test user... No action required...")
            #     # Backup data if enabled
            #     try:
            #         if get_config_value("IMAGEBACKUP").capitalize() == "True":
            #             backup_image(image_path)
            #     except Exception as e:
            #         logger.error(f"{e}")
            #         pass
            #     # Remove image after successful processing
            #     os.remove(image_path)
            #     logger.info(f"[+] Processed and deleted: {image_path}")
            #     continue

            # Get Yolo Config data from database
            YOLO_CONFIG_data = {}
            if YOLO_CONFIG:
                YOLO_CONFIG_data = json.loads(YOLO_CONFIG)

            try:
                if YOLO_CONFIG_data.get("conf", None) and YOLO_CONFIG_data.get("iou", None) and YOLO_CONFIG_data.get("imgsz", None):
                    CONF = float(YOLO_CONFIG_data["conf"])
                    IOU = float(YOLO_CONFIG_data["iou"])
                    IMGSZ = int(YOLO_CONFIG_data["imgsz"])
                else:
                    CONF, IOU, IMGSZ = 0.50, 0.40, 640
            except Exception as e:
                logger.error(f"Error getting yolo config data... Setting default value")
                CONF, IOU, IMGSZ = 0.50, 0.40, 640

            results = model.predict(img, save=False, save_crop=False, project=OUTPUT_FOLDER,
                                    name=image_name, exist_ok=True, conf=CONF, iou=IOU, imgsz=IMGSZ,
                                    verbose=False)

            if not results or len(results[0].boxes) == 0:
                logger.warning("No detections found.")
                show_error_popup(f"\nAnonymous ...")
                # Send image as bits to backup server through API
                # try:
                #     if get_config_value("IMAGEBACKUP").capitalize() == "True":
                #         backup_image(image_path)
                # except Exception as e:
                #     logger.error(f"{e}")
                #     pass
                # Remove image after successful processing
                os.remove(image_path)
                logger.info(f"[+] Processed and deleted: {image_path}")
                continue
            # Organize highest confidence box per class
            best_by_class = {}
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id not in best_by_class or conf > best_by_class[cls_id][1]:
                    best_by_class[cls_id] = (box, conf)

            crop_dir = os.path.join(OUTPUT_FOLDER, image_name.split('.')[0], "crops")
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir, exist_ok=True)

            class_image_paths = {}
            # Save the crops for each class
            for cls_id, (box, conf) in best_by_class.items():
                class_name = model.names[cls_id] if cls_id in model.names else str(cls_id)

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop_img = img[y1:y2, x1:x2]
                crop_file = os.path.join(crop_dir, f"{class_name}.jpg")

                # Save the crop
                cv2.imwrite(crop_file, crop_img)
                class_image_paths[class_name] = crop_file

                # class_image_paths.append(class_image_path)
                logger.info(f"Saved crop: {crop_file} with confidence {conf:.2f} - image: {class_name}")

            # Send image as bits to backup server through API
            # try:
            #     if get_config_value("IMAGEBACKUP").capitalize() == "True":
            #         backup_image(image_path)
            # except Exception as e:
            #     logger.error(f"{e}")
            #     pass

            # Remove image after successful processing
            os.remove(image_path)
            logger.info(f"[+] Processed and deleted: {os.path.basename(image_path)}")
            logger.info(f"[+] Yolo Time taken for {image_path}: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            # os.remove(image_path)
            logger.error(f"[!] Error processing {image_path}: {e}")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            if filename.lower().endswith(INPUT_EXTENSIONS):
                logger.info(f"[+] New image detected: {filename}")
                self.queue.put(event.src_path)
                logger.info(f"[+] Queued for processing: {filename}")

def clear_queue(q: multiprocessing.Queue):
    """Empty the queue safely."""
    while not q.empty():
        try:
            q.get_nowait()
        except:
            break


def main():
    image_queue = multiprocessing.Queue(maxsize=100)
    queued_files = set()

    # Start worker process
    worker = multiprocessing.Process(target=yolo_prediction, args=(image_queue,))
    worker.start()

    # Queue existing files
    for file in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, file)
        if os.path.isfile(file_path) and file.lower().endswith(INPUT_EXTENSIONS):
            image_queue.put(file_path)
            queued_files.add(file_path)

    # Start watchdog observer for new files
    event_handler = ImageHandler(image_queue)
    observer = Observer()
    observer.schedule(event_handler, INPUT_FOLDER, recursive=False)

    try:
        observer.start()
    except Exception as e:
        logger.error(f"Failed to start watchdog observer: {e}")

    logger.info(f"[+] Watcher Enabled to Folder: {INPUT_FOLDER}")

    last_rescan = time.time()
    RESCAN_INTERVAL = 60  # seconds

    logger.info(f"[INIT] Queue length: {image_queue.qsize() if not image_queue.empty() else 0}")

    while True:
        # try:
        #     if keyboard.is_pressed("esc"):
        #         logger.info("[!] ESC pressed. Stopping...")
        #         break
        # except:
        #     pass

        if int(time.time()) % 60 == 0:
            logger.info(f"[Monitor] Queue size: {image_queue.qsize()}")

        # Every minute, recheck folder for missed files
        if time.time() - last_rescan > RESCAN_INTERVAL:
            for file in os.listdir(INPUT_FOLDER):
                full_path = os.path.join(INPUT_FOLDER, file)
                if (
                    os.path.isfile(full_path)
                    and file.lower().endswith(INPUT_EXTENSIONS)
                    and full_path not in queued_files
                ):
                    image_queue.put(full_path)
                    queued_files.add(full_path)
                    logger.info(f"[Rescan] Added missed file: {file}")
            last_rescan = time.time()

        # Observer health check
        if not observer.is_alive():
            logger.warning("[!] Watchdog observer stopped. Restarting...")
            observer.stop()
            observer.join(timeout=2)
            observer = Observer()
            observer.schedule(event_handler, INPUT_FOLDER, recursive=False)
            observer.start()
            logger.info("[+] Watchdog observer restarted.")

        # Worker health check
        if not worker.is_alive():
            logger.warning("[!] Worker process died. Restarting...")
            worker = multiprocessing.Process(target=yolo_prediction, args=(image_queue,))
            worker.start()

        time.sleep(3)

    # ---- Shutdown ----
    observer.stop()
    observer.join()

    clear_queue(image_queue)
    image_queue.put(None)  # exit signal for worker
    if worker.is_alive():
        worker.join(timeout=5)
    logger.info("[+] All processes stopped and queue cleared.")
