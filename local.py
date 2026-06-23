import cv2
import numpy as np
import pyautogui
import math
from treys import Card, Evaluator
from ultralytics import YOLO

# --- Load BOTH YOLO models ---
# Make sure to update these paths once you train both models!
ZONE_MODEL_PATH = 'model/zone_best.pt' 
CARD_MODEL_PATH = 'model/card_m_1024_best.pt'

def scan_board_cascaded(img_rgb, zone_model, card_model, conf_zone=0.50, conf_card=0.40, debug=False):
    """
    Two-Stage Cascaded Pipeline:
    1. Detects Zones (Left Hand, Right Hand, Board).
    2. Crops the image to those exact zones.
    3. Scans the cropped images for Ranks and Suits.
    """
    print(f"\n--- STAGE 1: DETECTING ZONES ---")
    if img_rgb is None:
        return []
        
    h, w = img_rgb.shape[:2]
    
    # 1. Run the Zone Model on the full screen
    # Changed imgsz to 640 to perfectly match the resolution it was trained on!
    zone_results = zone_model(img_rgb, conf=conf_zone, imgsz=640, verbose=False)
    
    detected_zones = {}
    for result in zone_results:
        for box in result.boxes:
            zx1, zy1, zx2, zy2 = map(int, box.xyxy[0])
            label = zone_model.names[int(box.cls[0])].lower()
            
            # Save the coordinates of our 3 main zones
            if label in ['community_card', 'left_hand', 'right_hand']:
                detected_zones[label] = (zx1, zy1, zx2, zy2)

    print(f"Found {len(detected_zones)}/3 required zones.")
    if len(detected_zones) < 3:
        print("Waiting for all 3 zones to be visible...")
        return []

    all_cards = []
    
    # Define what the Card Model labels look like
    valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}
    valid_suits = {'S', 'H', 'D', 'C'}

    print(f"--- STAGE 2: READING CARDS ---")
    
    # 2. Loop through each detected zone, crop it, and pass it to the Card Model
    for zone_label, (zx1, zy1, zx2, zy2) in detected_zones.items():
        # --- NEW: Zone Padding ---
        # Push the walls out by 15 pixels so we don't chop off the top-left symbols!
        padding = 15
        p_zx1 = max(0, zx1 - padding)
        p_zy1 = max(0, zy1 - padding)
        p_zx2 = min(w, zx2 + padding)
        p_zy2 = min(h, zy2 + padding)
        
        # Crop the image safely using the padded coordinates
        zone_crop = img_rgb[p_zy1:p_zy2, p_zx1:p_zx2]
        crop_h, crop_w = zone_crop.shape[:2]
        
        # Run the Card Model on this specific tiny crop
        card_results = card_model(zone_crop, conf=conf_card, imgsz=1024, verbose=False)
        
        ranks = []
        suits = []
        
        # --- NEW: Print out the raw AI detections to see exactly what it's finding! ---
        raw_detections = [card_model.names[int(box.cls[0])].upper() for result in card_results for box in result.boxes]
        print(f"   [{zone_label}] Raw Detections: {raw_detections}")
        # ------------------------------------------------------------------------------
        
        for result in card_results:
            for box in result.boxes:
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                c_label = card_model.names[int(box.cls[0])].upper()
                
                # Coordinate center relative to the CROP
                ccx = (cx1 + cx2) // 2
                ccy = (cy1 + cy2) // 2
                
                item_data = {
                    'label': c_label,
                    'score': confidence,
                    'cx': ccx, 'cy': ccy,
                    'box': (cx1, cy1, cx2, cy2),
                    'used': False
                }
                
                if c_label in valid_ranks:
                    ranks.append(item_data)
                elif c_label in valid_suits:
                    suits.append(item_data)

        # 3. Pair Ranks and Suits (Geometry is now extremely simple since it's just the isolated zone)
        for r in ranks:
            best_suit = None
            min_dist = float('inf')
            
            for s in suits:
                if s['used']: continue
                
                # Suit must be roughly below the rank
                is_below = s['cy'] > (r['cy'] - 10)
                # --- FIX: Stricter Alignment ---
                # Reduced from 15% to 8% to prevent pairing a top-left Rank with a center Suit
                is_aligned = abs(s['cx'] - r['cx']) < (crop_w * 0.08)
                
                if is_below and is_aligned:
                    dist = math.hypot(r['cx'] - s['cx'], r['cy'] - s['cy'])
                    if dist < min_dist:
                        min_dist = dist
                        best_suit = s
            
            # If successfully paired
            if best_suit:
                best_suit['used'] = True
                suit_char = best_suit['label'][0]
                combined_label = r['label'] + suit_char 
                
                # 4. CRITICAL: Translate coordinates back to the GLOBAL screen
                # Translated using padded zone variables
                global_x1 = min(r['box'][0], best_suit['box'][0]) + p_zx1
                global_y1 = min(r['box'][1], best_suit['box'][1]) + p_zy1
                global_x2 = max(r['box'][2], best_suit['box'][2]) + p_zx1
                global_y2 = max(r['box'][3], best_suit['box'][3]) + p_zy1
                
                global_cx = (global_x1 + global_x2) // 2
                global_cy = (global_y1 + global_y2) // 2
                
                # Map zone label to our evaluator's required format
                target_zone = 'board' if zone_label == 'community_card' else ('left' if zone_label == 'left_hand' else 'right')
                
                all_cards.append({
                    'label': combined_label,
                    'score': (r['score'] + best_suit['score']) / 2, 
                    'cx': global_cx,
                    'cy': global_cy,
                    'zone': target_zone,
                    'box': (global_x1, global_y1, global_x2, global_y2)
                })

    # Sort cards left-to-right based on X coordinate
    matches = sorted(all_cards, key=lambda x: x['cx'])

    if debug:
        debug_img = img_rgb.copy()
        
        # Draw Zones
        for z_label, (zx1, zy1, zx2, zy2) in detected_zones.items():
            cv2.rectangle(debug_img, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(debug_img, z_label, (zx1, zy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # Draw Cards
        for m in matches:
            gx1, gy1, gx2, gy2 = m['box']
            color = (0, 255, 0) if m['zone'] == 'board' else (255, 0, 0)
            cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(debug_img, f"{m['label']} {m['score']:.2f}", (gx1, gy1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # ALWAYS save to file instead of trying to force a buggy pop-up window
        cv2.imwrite("debug_cascaded_output.jpg", debug_img)
        print(">>> SAVED 'debug_cascaded_output.jpg' - Open this file to see what the bot saw!")
        
    return matches

def normalize_card(label):
    label = label.upper()
    if len(label) == 3 and label.startswith('10'):
        return 'T' + label[-1].lower()
    elif len(label) == 2:
        return label[0] + label[1].lower()
    return label

def evaluate_and_click(matches, offset_x=0, offset_y=0):
    print("\n--- EVALUATING HANDS ---")
    board_matches = [m for m in matches if m['zone'] == 'board']
    left_matches = [m for m in matches if m['zone'] == 'left']
    right_matches = [m for m in matches if m['zone'] == 'right']

    if len(board_matches) != 5 or len(left_matches) != 2 or len(right_matches) != 2:
        print(f"Warning: Incorrect card counts. Board: {len(board_matches)}, Left: {len(left_matches)}, Right: {len(right_matches)}")
        return False

    evaluator = Evaluator()
    try:
        board = [Card.new(normalize_card(m['label'])) for m in board_matches]
        left_hand = [Card.new(normalize_card(m['label'])) for m in left_matches]
        right_hand = [Card.new(normalize_card(m['label'])) for m in right_matches]
    except Exception as e:
        print(f"Error parsing cards: {e}")
        return False

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
        print("Winner: TIE")
        winning_matches = left_matches

    target_x = sum(m['cx'] for m in winning_matches) / len(winning_matches)
    target_y = sum(m['cy'] for m in winning_matches) / len(winning_matches)

    final_x = int(target_x + offset_x)
    final_y = int(target_y + offset_y)

    print(f"Action: Clicking coordinates X:{final_x}, Y:{final_y}")
    pyautogui.click(x=final_x, y=final_y)
    return True

if __name__ == "__main__":
    import mss
    import time
    import keyboard

    pyautogui.FAILSAFE = True 
    
    try:
        print("Loading AI Models... (This might take a second)")
        zone_model = YOLO(ZONE_MODEL_PATH)
        card_model = YOLO(CARD_MODEL_PATH)
        print("Both models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Ensure 'zone_best.pt' and 'card_best_s.pt' exist!")
        exit(1)
    
    print("Script ready. Switch to your Chrome window!")
    print("Press [R] to execute a single round (Scan & Click).")
    print("Press [Q] at any time to terminate.")

    with mss.MSS() as sct:
        monitor = sct.monitors[1] 
        
        while True:
            if keyboard.is_pressed('q'):
                print("\nQ pressed. Terminating script.")
                break

            if keyboard.is_pressed('r'):
                print("\n--- R Pressed: Executing Round ---")
                
                sct_img = sct.grab(monitor)
                img_rgb = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                
                # Execute the cascaded scan on the full raw screenshot
                # Lowered conf_card to 0.10
                matches = scan_board_cascaded(img_rgb, zone_model, card_model, conf_zone=0.25, conf_card=0.10, debug=True) 
                
                if matches and len(matches) == 9:
                    # Pass monitor position for PyAutoGUI
                    evaluate_and_click(matches, offset_x=monitor['left'], offset_y=monitor['top'])
                else:
                    if matches is not None and len(matches) > 0:
                        print(f"Stage 2 Incomplete: Paired {len(matches)}/9 cards. Check debug image!")
                
                time.sleep(0.5) # Prevent multiple triggers from a single key press
            
            time.sleep(0.01) # Small sleep to prevent CPU hogging in the while loop