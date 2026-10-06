"""Create a layout preview from the two finalized Fig. 5 components.

Run: python paper/figures/fig5/assemble_fig5.py
Only writes fig5_layout_preview.png beside this file; source PNGs stay unchanged.
This is a layout aid, not a replacement for the finalized A/B images.
"""

from pathlib import Path
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
FILES = [HERE / "fig5A_metadrive_platform.png", HERE / "fig5B_six_panel_cases.png"]
OUT = HERE / "fig5_layout_preview.png"


def main():
    images = []
    for filename in FILES:
        with Image.open(filename) as original:
            images.append(original.convert("RGB"))
    target_width = max(image.width for image in images)
    scaled = []
    for image in images:
        if image.width != target_width:
            height = round(image.height * target_width / image.width)
            image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        scaled.append(image)
    gap = round(target_width * 0.025)
    page = Image.new("RGB", (target_width, sum(x.height for x in scaled) + gap), "white")
    y = 0
    for image in scaled:
        page.paste(image, (0, y))
        y += image.height + gap
    page.save(OUT, optimize=True)
    print(f"Layout preview: {OUT}")
    print("The two finalized component PNGs were read only.")


if __name__ == "__main__":
    main()
