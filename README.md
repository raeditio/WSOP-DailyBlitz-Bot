# PokerVisionBot — WSOP Daily Blitz Assistant

PokerVisionBot is a small utility that automatically reads WSOP Daily Blitz poker tables, evaluates both hands against
the community cards, and clicks the winning hand. It combines two YOLO object-detection models (zone detection +
card detection) with a poker hand evaluator to make decisions.

## Highlights

- Two-stage cascaded detection: locate zones (community, left, right) then read ranks and suits inside each zone
- Geometric pairing logic to match rank and suit detections reliably
- Uses the `treys` evaluator for accurate poker hand ranking
- Saves debug images (`debug_cascaded_output.jpg`) showing detected zones and cards

### Releases

Packaged distributions (installers and prebuilt binaries) are published on the project's GitHub Releases page. Check
the Releases tab to download the latest release assets instead of building from source.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the trained models are present in the `model/` directory:

- `model/zone_best.pt` (zone detector, trained at 640px)
- `model/card_m_1024_best.pt` (card detector, trained at 1024px)

3. Run the bot and follow on-screen instructions:

```bash
python local.py
```

Then switch to your browser with WSOP Daily Blitz in full-screen mode. The bot will automatically begin scanning and
clicking winning hands once it detects the game
Press `Q` or close the app to quit the program.

## Usage & Controls

- **Automatic** — Starts scanning and clicking when a WSOP Daily Blitz table is detected on screen
- `Q` — Terminate the program

Notes:
- The bot captures the full screen. Run the game in a single, dedicated display to avoid false detections.
- PyAutoGUI failsafe is enabled: move your mouse to a screen corner to abort if needed.

## Implementation Details

- Zone model input size: 640px
- Card model input size: 1024px
- Zone padding: 15 px (prevents cropping important card symbols)
- Suit alignment tolerance: 8% of zone crop width (strict horizontal alignment for pairing)
- Default detection thresholds used by the script: zone conf=0.25, card conf=0.10

Model files and thresholds can be tuned in `local.py` by updating the constants near the top of the file.

## Debugging

When run with debug enabled the script writes `debug_cascaded_output.jpg` to the working directory. Open this
image to verify detected zones, paired cards, and bounding boxes.

If the pipeline misses cards or pairs them incorrectly, try:

- Increasing the card detection confidence threshold
- Ensuring the game UI is unobstructed and consistent with training data (fonts, scale, card artwork)
- Re-training or fine-tuning the card model with additional labeled images from your environment

## Troubleshooting

- Incorrect card counts: ensure the entire table is visible and models are loaded without errors
- Models fail to load: confirm `ultralytics` is installed and model files exist at the paths above
- Clicking the wrong location: verify monitor offsets (the script uses the screenshot monitor offsets when clicking)

## Contributing

Contributions and improvements are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests if applicable
4. Open a pull request describing your change

## Safety & Ethics

This tool automates interactions with a third-party game. Use it only where allowed by the game's terms of service
and for personal/experimental purposes. The author takes no responsibility for misuse.