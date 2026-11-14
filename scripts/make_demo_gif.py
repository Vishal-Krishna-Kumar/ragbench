"""Render a terminal demo GIF from the real output of a command.

The command is run, its output is captured, and the frames are drawn with Pillow: the command
is typed out, then the output appears line by line. Nothing in the GIF is written by hand.

    pip install pillow
    python scripts/make_demo_gif.py                      # SciFact run -> docs/demo.gif
    python scripts/make_demo_gif.py --command "ragbench data/sample --format table" --out docs/sample.gif
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (13, 17, 23)
TITLE_BAR = (22, 27, 34)
TEXT = (201, 209, 217)
DIM = (139, 148, 158)
PROMPT = (63, 185, 80)
GOOD = (63, 185, 80)
BAD = (248, 81, 73)
WARN = (210, 153, 34)
DOTS = ((255, 95, 86), (255, 189, 46), (39, 201, 63))

FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def run(command: str) -> list[str]:
    """Run the command and return its output lines, standard error first."""
    arguments = shlex.split(command)
    if arguments[0] == "ragbench":
        arguments = [sys.executable, "-m", "ragbench.cli", *arguments[1:]]
    done = subprocess.run(arguments, capture_output=True, text=True, check=True)
    return [*done.stderr.splitlines(), *done.stdout.splitlines()]


def split_colour(line: str, header: str) -> tuple[int, int, tuple[int, int, int]] | None:
    """Character range and colour for the "Answer split" value of a table row."""
    column = header.find("Answer split")
    if column == -1 or "%" not in line or len(line) < column + 12:
        return None
    cell = line[column : column + 12]
    try:
        value = float(cell.strip().rstrip("%"))
    except ValueError:
        return None
    colour = GOOD if value < 2 else WARN if value < 10 else BAD
    start = column + len(cell) - len(cell.lstrip())
    return start, column + 12, colour


def render(command: str, lines: list[str], out: Path, font_size: int = 17) -> None:
    font = load_font(font_size)
    width_of = font.getlength("M")
    line_height = int(font_size * 1.55)
    padding, bar = 22, 38
    columns = max(len(f"$ {command}"), *(len(line) for line in lines))
    width = int(columns * width_of) + 2 * padding
    height = bar + 2 * padding + line_height * (len(lines) + 2)
    header = next((line for line in lines if "Answer split" in line), "")

    def frame(typed: int, shown: int, cursor: bool) -> Image.Image:
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, bar), fill=TITLE_BAR)
        for number, colour in enumerate(DOTS):
            x = 18 + number * 22
            draw.ellipse((x, 12, x + 13, 25), fill=colour)
        y = bar + padding
        draw.text((padding, y), "$", font=font, fill=PROMPT)
        draw.text((padding + 2 * width_of, y), command[:typed], font=font, fill=TEXT)
        if cursor:
            x = padding + (2 + typed) * width_of
            draw.rectangle((x, y + 2, x + width_of - 1, y + font_size + 3), fill=TEXT)
        y += 2 * line_height
        for line in lines[:shown]:
            dim = line.startswith("Dataset:") or set(line) <= {"-", " "}
            draw.text((padding, y), line, font=font, fill=DIM if dim else TEXT)
            highlight = split_colour(line, header)
            if highlight:
                start, end, colour = highlight
                x = padding + start * width_of
                draw.rectangle((x, y, padding + end * width_of, y + line_height - 4), BACKGROUND)
                draw.text((x, y), line[start:end], font=font, fill=colour)
            y += line_height
        return image

    frames: list[Image.Image] = []
    durations: list[int] = []

    def add(image: Image.Image, milliseconds: int) -> None:
        frames.append(image)
        durations.append(milliseconds)

    add(frame(0, 0, True), 700)
    for typed in range(1, len(command) + 1):
        add(frame(typed, 0, True), 55)
    add(frame(len(command), 0, True), 500)
    for shown in range(1, len(lines) + 1):
        add(frame(len(command), shown, False), 220 if shown > 3 else 120)
    add(frame(len(command), len(lines), False), 6000)

    out.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[-1].quantize(colors=48, dither=Image.Dither.NONE)
    converted = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in frames]
    converted[0].save(
        out, save_all=True, append_images=converted[1:], duration=durations, loop=0, optimize=True
    )
    frames[-1].save(out.with_suffix(".png"))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(frames)} frames, {width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--command", default="ragbench data/scifact --format table")
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    args = parser.parse_args()
    render(args.command, run(args.command), args.out)


if __name__ == "__main__":
    main()
