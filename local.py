import cv2
import numpy as np
import pyautogui
from treys import Card, Evaluator
from ultralytics import YOLO # pip install ultralytics

# Load your trained YOLO model here (Updated to match the yolo26 folder!)
MODEL_PATH = 'model/best.pt' 

def scan_board_yolo(img_rgb, model, conf_threshold=0.60, debug=False):
    """
    Scans the board using a trained YOLO object detection model.
    It is extremely fast and completely immune to minor pixel/scaling differences.
    """
    print(f"\n--- SCANNING BOARD (YOLO) ---")
    if img_rgb is None:
        return []
        
    h, w = img_rgb.shape[:2]
    
    # Run inference on the image
    results = model(img_rgb, conf=conf_threshold, verbose=False)
    
    matches = []
    
    # Extract bounding boxes and labels
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates, confidence, and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id] # e.g., 'AS', '10H'
            
            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Determine Zone based on screen coordinates rather than cropped ROIs
            if cy < h * 0.45:
                zone_name = 'board'
            elif cx < w * 0.50:
                zone_name = 'left'
            else:
                zone_name = 'right'
                
            matches.append({
                'label': label,
                'score': confidence,
                'cx': cx,
                'cy': cy,
                'zone': zone_name,
                'box': (x1, y1, x2, y2)
            })

    # Sort matches left-to-right based on X coordinate for cleaner evaluation
    matches = sorted(matches, key=lambda x: x['cx'])

    if debug:
        debug_img = img_rgb.copy()
        for m in matches:
            x1, y1, x2, y2 = m['box']
            color = (0, 255, 0) if m['zone'] == 'board' else (255, 0, 0)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, f"{m['label']} {m['score']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        try:
            cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO Detections", debug_img)
            print(f"Found {len(matches)} cards. Click image window and press any key...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print(f"Found {len(matches)} cards.")
            print("Notice: OpenCV GUI not supported (headless version installed).")
            cv2.imwrite("debug_yolo_output.jpg", debug_img)
            print("Saved debug image to 'debug_yolo_output.jpg' instead.")
        
    return matches

def normalize_card(label):
    """Converts Roboflow dataset format (e.g. '10H', 'AS') to Treys library standard ('Th', 'As')"""
    label = label.upper()
    
    # Handle '10' becoming 'T' and lowercase the suit
    if len(label) == 3 and label.startswith('10'):
        return 'T' + label[-1].lower()
    elif len(label) == 2:
        return label[0] + label[1].lower()
        
    return label

def evaluate_and_click(matches):
    """Evaluates the hands and clicks the winning one."""
    print("\n--- EVALUATING HANDS ---")
    board_matches = [m for m in matches if m['zone'] == 'board']
    left_matches = [m for m in matches if m['zone'] == 'left']
    right_matches = [m for m in matches if m['zone'] == 'right']

    if len(board_matches) != 5:
        print(f"Warning: Found {len(board_matches)} board cards instead of 5.")
        return False
    if len(left_matches) != 2 or len(right_matches) != 2:
        print(f"Warning: Found {len(left_matches)} left cards and {len(right_matches)} right cards.")
        return False

    evaluator = Evaluator()
    
    try:
        board = [Card.new(normalize_card(m['label'])) for m in board_matches]
        left_hand = [Card.new(normalize_card(m['label'])) for m in left_matches]
        right_hand = [Card.new(normalize_card(m['label'])) for m in right_matches]
    except Exception as e:
        print(f"Error parsing cards: {e}. Ensure model labels map correctly to Treys format (Ac, Th).")
        return

    left_score = evaluator.evaluate(board, left_hand)
    right_score = evaluator.evaluate(board, right_hand)

    print(f"Left Hand: {evaluator.class_to_string(evaluator.get_rank_class(left_score))} (Score: {left_score})")
    print(f"Right Hand: {evaluator.class_to_string(evaluator.get_rank_class(right_score))} (Score: {right_score})")

    if left_score < right_score:
        print("Winner: LEFT HAND")
        winning_matches = left_matches
    elif right_score < left_score:
        print("Winner: RIGHT HAND")
        winning_matches = right_matches
    else:
        print("Winner: TIE (Clicking Left as default)")
        winning_matches = left_matches

    target_x = sum(m['cx'] for m in winning_matches) / len(winning_matches)
    target_y = sum(m['cy'] for m in winning_matches) / len(winning_matches)

    print(f"Action: Clicking coordinates X:{int(target_x)}, Y:{int(target_y)}")
    pyautogui.click(x=int(target_x), y=int(target_y))
    return True

if __name__ == "__main__":
    import mss
    import time
    import keyboard

    pyautogui.FAILSAFE = True 
    
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded YOLO model: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("You must train a YOLO26 model and place 'best.pt' in the same folder.")
        exit(1)
    
    print("Script ready. Switch to your Chrome window!")
    print("Press [R] to execute a single round (Scan & Click).")
    print("Press [Q] to terminate the script.")

    with mss.MSS() as sct:
        monitor = sct.monitors[1] 
        
        while True:
            if keyboard.is_pressed('q'):
                print("Q pressed. Terminating script.")
                break

            if keyboard.is_pressed('r'):
                print("\n--- R Pressed: Executing Round ---")
                
                sct_img = sct.grab(monitor)
                img_rgb = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                
                # Use YOLO scan with a lower threshold to see if it's catching ANYTHING
                matches = scan_board_yolo(img_rgb, model, conf_threshold=0.25, debug=True) 
                
                if matches and len(matches) == 9:
                    evaluate_and_click(matches)
                else:
                    print(f"Required 9 cards, but found {len(matches) if matches else 0}.")
                    print("Check 'debug_yolo_output.jpg' to see what the bot is looking at!")
                
                time.sleep(0.5)
            
            time.sleep(0.01)