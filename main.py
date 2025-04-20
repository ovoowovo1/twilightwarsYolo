import threading
from ultralytics import YOLO
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import keyboard
import math
import mss
import time
import traceback # 导入 traceback 模块
import logging

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- 加载模型 ---
try:
    logger.info("Loading YOLO model...")
    # model = YOLO(r"model\model_- 31 august 2024 2_41.onnx", task='detect')
    model = YOLO(r"model\best.engine")
    logger.info("YOLO model loaded successfully.")
except FileNotFoundError:
    logger.critical(f"Model file not found at path: model\\model_- 19 april 2025 16_02.onnx")
    exit()
except Exception as e:
    logger.critical(f"Fatal Error loading YOLO model: {e}")
    logger.critical(traceback.format_exc())
    exit()

# --- 全局变量 ---
me_position = None
nearest_enemy_position = None
enemies = []
frame_for_display = None
lock = threading.Lock()
running = True

# --- 函数定义 (get_browser_window, find_nearest_enemy) ---
# (Keep the get_browser_window and find_nearest_enemy functions as before)
def get_browser_window():
    """查找并返回一个可见的浏览器窗口对象"""
    browser_titles = ['Chrome', 'Firefox', 'Edge', 'Safari', 'Opera', 'Brave']
    active_window = gw.getActiveWindow()
    if active_window:
        try:
            if any(title.lower() in active_window.title.lower() for title in browser_titles):
                if active_window.visible and not active_window.isMinimized and active_window.width > 100 and active_window.height > 100:
                    logger.info(f"Using active window: {active_window.title}")
                    return active_window
                else:
                    logger.info(f"Active window '{active_window.title}' is a browser but not suitable.")
            else:
                 logger.info(f"Active window '{active_window.title}' is not a recognized browser.")
        except Exception as e_active:
             logger.warning(f"Error checking active window: {e_active}")

    logger.info("Searching all windows for a suitable browser...")
    all_windows = gw.getAllWindows()
    valid_browsers = []
    for window in all_windows:
        try:
            if (any(title.lower() in window.title.lower() for title in browser_titles) and
                    window.visible and not window.isMinimized and
                    window.width > 100 and window.height > 100):
                valid_browsers.append(window)
        except Exception:
            pass

    if valid_browsers:
        target_window = max(valid_browsers, key=lambda w: w.width * w.height)
        logger.info(f"Found suitable browser window: {target_window.title} (Size: {target_window.width}x{target_window.height})")
        return target_window
    else:
        logger.error("No suitable browser window found after searching all windows.")
        return None

def find_nearest_enemy(me_pos, enemy_list):
    """计算并返回离 me_pos 最近的敌人坐标"""
    if not enemy_list or me_pos is None:
        return None
    try:
        if not (isinstance(me_pos, (tuple, list)) and len(me_pos) == 2 and all(isinstance(n, (int, float)) for n in me_pos)):
             logger.warning(f"Invalid me_pos format: {me_pos}")
             return None
        valid_enemies = [e for e in enemy_list if isinstance(e, (tuple, list)) and len(e) == 2 and all(isinstance(n, (int, float)) for n in e)]
        if not valid_enemies:
             return None
        if hasattr(math, 'dist'):
             nearest = min(valid_enemies, key=lambda e: math.dist(me_pos, e))
        else:
             nearest = min(valid_enemies, key=lambda e: math.hypot(me_pos[0] - e[0], me_pos[1] - e[1]))
        return nearest
    except Exception as e:
        logger.error(f"Error in find_nearest_enemy: {e}")
        logger.error(traceback.format_exc())
        return None


# --- 捕获与检测线程 ---
def capture_and_detect(browser_window):
    global me_position, nearest_enemy_position, enemies, frame_for_display, running, lock
    logger.info("Capture and detect thread started.")

    # --- 可调整参数 ---
    inference_size = 640
    confidence_threshold = 0.4
    max_box_area_ratio = 0.10
    iou_threshold = 0.45
    device = 'cuda:0'
    # --- 修改点: 降低目标 FPS ---
    target_fps = 180 # Reduced from 30 (or previous value) to 15 FPS
    # ---------------------------
    logger.info(f"Detection target FPS set to: {target_fps}")

    try:
        with mss.mss() as sct:
            monitor = {}
            process_width = 0
            process_height = 0
            first_run = True

            while running:
                loop_start_time = time.perf_counter()
                try:
                    # --- 检查和更新窗口信息 ---
                    try:
                        if not browser_window.visible:
                            logger.warning("Browser window no longer visible. Stopping thread.")
                            running = False
                            break
                        current_geo = browser_window.box
                        monitor = {
                            "top": current_geo.top, "left": current_geo.left,
                            "width": current_geo.width, "height": current_geo.height
                        }
                        if monitor["width"] <= 0 or monitor["height"] <= 0:
                            logger.warning(f"Browser window invalid size. Waiting...")
                            time.sleep(1); continue
                        process_width = monitor["width"] // 2
                        process_height = monitor["height"] // 2
                        if process_width <= 0 or process_height <= 0:
                            logger.warning("Processing size invalid. Skipping."); time.sleep(0.5); continue
                        if first_run:
                             logger.info(f"Monitor region: {monitor}, Processing res: {process_width}x{process_height}")
                             first_run = False
                    except gw.PyGetWindowException:
                        logger.warning("Browser window lost (GWExc). Stopping."); running = False; break
                    except AttributeError as e_attr:
                         logger.error(f"Attr error win props: {e_attr}"); logger.error(traceback.format_exc()); running = False; break
                    except Exception as e_win:
                        logger.error(f"Error updating window info: {e_win}"); time.sleep(1); continue
                    # ---------------------------

                    # 1. Capture Screen
                    # logger.debug("Capturing screen.") # DEBUG level is verbose
                    try:
                        screenshot = sct.grab(monitor)
                        raw_frame = np.array(screenshot)
                    except Exception as e_grab:
                        logger.error(f"Screen capture error: {e_grab}"); time.sleep(1); continue

                    # 2. Preprocess Frame
                    if raw_frame.size == 0: logger.warning("Empty frame captured."); continue
                    # logger.debug("Preprocessing frame.")
                    frame_resized = cv2.resize(raw_frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
                    if frame_resized.shape[2] == 4: current_frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_BGRA2BGR)
                    elif frame_resized.shape[2] == 3: current_frame_bgr = frame_resized
                    else: logger.warning(f"Unexpected channels: {frame_resized.shape[2]}."); continue

                    # 3. Run YOLO Detection
                    # logger.debug("Running YOLO detection...")
                    try:
                        results = model(current_frame_bgr, imgsz=inference_size, conf=confidence_threshold, iou=iou_threshold, device=device, verbose=False)
                        #speed_info = results[0].speed
                        # 你可以打印整个字典，或者只打印推理时间
                        #logger.info(f"Speed Info: {speed_info}")
                    except Exception as e_yolo:
                        logger.error(f"YOLO detection error: {e_yolo}"); logger.error(traceback.format_exc()); time.sleep(0.1); continue

                    # 4. Process Results
                    # logger.debug("Processing results.")
                    local_me_position = None; local_enemies = []; detected_boxes_for_drawing = []
                    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes.cpu().numpy(); names = results[0].names
                        frame_area = process_width * process_height; frame_area = 1 if frame_area <= 0 else frame_area
                        for box in boxes:
                            try:
                                x1, y1, x2, y2 = map(int, box.xyxy[0]); cls_id = int(box.cls[0]); conf = float(box.conf[0]); class_name = names.get(cls_id, f'ID_{cls_id}')
                                box_width = x2 - x1; box_height = y2 - y1
                                if box_width <= 0 or box_height <= 0: continue
                                box_area = box_width * box_height
                                if max_box_area_ratio is not None and (box_area / frame_area > max_box_area_ratio): continue
                                center_x = (x1 + x2) // 2; center_y = (y1 + y2) // 2
                                # !!! CHECK YOUR CLASS NAMES !!!
                                if class_name == 'Me':
                                    if local_me_position is None: local_me_position = (center_x, center_y)
                                    detected_boxes_for_drawing.append(((x1, y1, x2, y2), 'Me', (0, 255, 0)))
                                elif class_name == 'Emeny': # OR 'Enemy' ?? Check your model training!
                                    local_enemies.append((center_x, center_y))
                                    detected_boxes_for_drawing.append(((x1, y1, x2, y2), 'Enemy', (0, 0, 255)))
                            except Exception as e_box_proc: logger.warning(f"Error processing box: {e_box_proc}"); continue
                    # logger.debug(f"Processed: Me={local_me_position}, Enemies={len(local_enemies)}")

                    # 5. Find Nearest Enemy
                    local_nearest_enemy = find_nearest_enemy(local_me_position, local_enemies)

                    # 6. Update Global Variables & Draw Frame
                    # logger.debug("Updating globals & drawing.")
                    with lock:
                        me_position = local_me_position; enemies = local_enemies; nearest_enemy_position = local_nearest_enemy
                        display_frame = current_frame_bgr.copy()
                        for box_coords, name, color in detected_boxes_for_drawing:
                            x1, y1, x2, y2 = box_coords; cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        if local_nearest_enemy and local_me_position:
                            cv2.circle(display_frame, local_nearest_enemy, 10, (255, 0, 0), 3)
                            cv2.line(display_frame, local_me_position, local_nearest_enemy, (255, 255, 0), 2)
                        frame_for_display = display_frame

                    # --- Timing and Sleep ---
                    loop_end_time = time.perf_counter(); processing_time = loop_end_time - loop_start_time
                    #logger.info(f"Loop time: {processing_time:.4f}s")
                    time.sleep(0.001)

                except Exception as e_loop:
                    logger.error(f"Unexpected error in loop iteration: {e_loop}"); logger.error(traceback.format_exc()); time.sleep(1)
    except Exception as e_thread:
        logger.critical(f"Critical error in capture_and_detect thread: {e_thread}"); logger.critical(traceback.format_exc()); running = False
    logger.info("Capture and detect thread finished.")


# --- 用户输入与动作线程 ---
def handle_user_input(browser_window):
    global me_position, nearest_enemy_position, running, lock
    logger.info("Input handler thread started.")
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    bias_ratio = 0.3
    try:
        while running:
            try:
                # --- 检查窗口有效性 ---
                try:
                    if not browser_window.visible:
                        logger.warning("Browser window no longer visible. Stopping thread."); running = False; break
                except gw.PyGetWindowException:
                    logger.warning("Browser window lost (GWExc). Stopping."); running = False; break
                except AttributeError as e_attr:
                     logger.error(f"Attr error win props: {e_attr}"); logger.error(traceback.format_exc()); running = False; break
                except Exception as e_win_input:
                     logger.error(f"Error checking window: {e_win_input}"); time.sleep(1); continue
                # -----------------------

                if keyboard.is_pressed('k'):
                    # logger.debug("'k' key pressed.") # Can be spammy
                    t_press_detected = time.perf_counter() # 记录检测到按键的时间
                    
                    current_me_pos = None; current_enemy_pos = None; window_left = 0; window_top = 0
                    try:
                         if not browser_window.visible: continue
                         window_left = browser_window.left; window_top = browser_window.top
                    except Exception as e_get_pos: logger.warning(f"Could not get win pos: {e_get_pos}"); continue

                    # logger.debug("Acquiring lock.")
                    with lock:
                        current_me_pos = me_position; current_enemy_pos = nearest_enemy_position
                    # logger.debug(f"Positions: Me={current_me_pos}, Enemy={current_enemy_pos}")

                    if current_me_pos and current_enemy_pos:
                        try:
                            me_x_res, me_y_res = current_me_pos; enemy_x_res, enemy_y_res = current_enemy_pos
                            if not (isinstance(me_x_res, (int, float)) and isinstance(me_y_res, (int, float)) and isinstance(enemy_x_res, (int, float)) and isinstance(enemy_y_res, (int, float))):
                                logger.warning(f"Invalid coord types: Me={current_me_pos}, Enemy={current_enemy_pos}"); continue
                            
                            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                            # --- 修改点: 使用线性插值计算靠近“我”的目标点 ---
                            # 公式: target_x = me_x + ratio * (enemy_x - me_x)
                            target_x_res = me_x_res + bias_ratio * (enemy_x_res - me_x_res)
                            target_y_res = me_y_res + bias_ratio * (enemy_y_res - me_y_res)
                            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                            # --- 使用新的 target_x_res, target_y_res 计算屏幕坐标 ---
                            screen_x = window_left + int(target_x_res * 2) # 使用 target_x_res
                            screen_y = window_top + int(target_y_res * 2) # 使用 target_y_res
                            logger.debug(f"Calculated biased screen target: ({screen_x}, {screen_y})")
                            # ---
                            # logger.debug(f"Midpoint screen: ({screen_x}, {screen_y})")

                            screen_w, screen_h = pyautogui.size()
                            if 0 <= screen_x < screen_w and 0 <= screen_y < screen_h:
                                t_before_click = time.perf_counter() 
                                
                                # logger.debug("Clicking midpoint.")
                                pyautogui.moveTo(screen_x, screen_y, duration=0); pyautogui.click()
                                t_after_click = time.perf_counter()  # 点击后时间
                                
                                # --- 计算并记录延迟 ---
                                action_latency_ms = (t_after_click - t_press_detected) * 1000 # 转换为毫秒
                                click_duration_ms = (t_after_click - t_before_click) * 1000 # 单独看点击耗时
                                logger.info(f"Clicked biased target at ({screen_x}, {screen_y}). "
                                            f"Action Latency: {action_latency_ms:.2f} ms "
                                            f"(Click itself: {click_duration_ms:.2f} ms)")
                                # ---------------------
                                time.sleep(0.001) # Post-click delay
                            else: logger.warning(f"Click pos outside screen: ({screen_x}, {screen_y})")
                        except pyautogui.FailSafeException: logger.critical("PyAutoGUI FailSafe triggered!"); running = False; break
                        except Exception as e_mouse: logger.error(f"Mouse action error: {e_mouse}"); logger.error(traceback.format_exc())
                    # else: logger.debug("K pressed, Me/Enemy not found.")

                # --- 修改点: 增加输入循环的 sleep 时间 ---
                time.sleep(0.001) # Increased from 0.01 to reduce polling frequency
                # ------------------------------------

            except KeyboardInterrupt: logger.info("Ctrl+C in input handler."); running = False; break
            except Exception as e_loop_input: logger.error(f"Unexpected error in input loop: {e_loop_input}"); logger.error(traceback.format_exc()); time.sleep(1)
    except Exception as e_thread_input: logger.critical(f"Critical error in input thread: {e_thread_input}"); logger.critical(traceback.format_exc()); running = False
    logger.info("Input handler thread finished.")


# --- 主执行部分 ---
if __name__ == "__main__":
    logger.info("Script starting...")
    browser_window = get_browser_window()
    if browser_window is None: logger.critical("Exiting: No browser window found."); exit()
    else:
        try:
             win_info = f"Target: '{browser_window.title}' ({browser_window.left},{browser_window.top}) {browser_window.width}x{browser_window.height}"
             logger.info(win_info)
             if browser_window.width <= 0 or browser_window.height <= 0: logger.critical("Win invalid size."); exit()
        except Exception as e_init_win: logger.critical(f"Failed get win info: {e_init_win}"); logger.critical(traceback.format_exc()); exit()

    threads = [
        threading.Thread(target=capture_and_detect, args=(browser_window,), name="CaptureDetectThread", daemon=True),
        threading.Thread(target=handle_user_input, args=(browser_window,), name="InputHandlerThread", daemon=True)
    ]
    logger.info("Starting threads...")
    for thread in threads: thread.start()

    logger.info("Main loop started. Press 'q' in OpenCV window to exit.")
    try:
        while running:
            display_this_frame = None
            # logger.debug("Main loop: Getting frame.")
            with lock:
                if frame_for_display is not None:
                    if isinstance(frame_for_display, np.ndarray) and frame_for_display.size > 0: display_this_frame = frame_for_display.copy()
                    else: frame_for_display = None
            # logger.debug("Main loop: Frame obtained.")

            if display_this_frame is not None:
                try:
                    # logger.debug("Main loop: Displaying frame.")
                    cv2.imshow("YOLO Real-time Browser Detection", display_this_frame)
                except Exception as e_show: logger.error(f"cv2.imshow error: {e_show}"); cv2.destroyAllWindows(); time.sleep(0.1) # Try destroy/recreate
            # else: logger.debug("No frame to display.")

            try:
                # logger.debug("Main loop: Checking quit key.")
                key = cv2.waitKey(33) # Wait ~30ms (limits display loop rate)
                if key != -1:
                    if key & 0xFF == ord('p'): logger.info("Quit key 'p' pressed."); running = False; break
                    # else: logger.debug(f"Key pressed: {key}")
            except Exception as e_wait: logger.error(f"cv2.waitKey error: {e_wait}"); running = False; break

    except KeyboardInterrupt: logger.info("Ctrl+C detected in main thread."); running = False
    except Exception as e_main: logger.critical(f"Critical error in main loop: {e_main}"); logger.critical(traceback.format_exc()); running = False
    finally:
        logger.info("Cleaning up..."); running = False
        cv2.destroyAllWindows()
        logger.info("Waiting briefly for threads..."); time.sleep(0.5)
        logger.info("Program finished.")