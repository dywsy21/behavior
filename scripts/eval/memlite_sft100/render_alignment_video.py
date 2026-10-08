"""Render rollout and chunk-review videos from an AlignmentTrace sidecar."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import textwrap

import av
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
LEFT_WIDTH = 960
PANEL_WIDTH = WIDTH - LEFT_WIDTH
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_SIZE = 17
LINE_HEIGHT = 21
PANEL_MARGIN = 20
LINES_PER_PAGE = 45


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    expected = list(range(len(rows)))
    chunks = [row["chunk_id"] for row in rows]
    if chunks != expected:
        raise ValueError(f"Non-contiguous chunks in {path}: {chunks[:5]} ... {chunks[-5:]}")
    if [row["control_step"] for row in rows] != [chunk * 16 for chunk in expected]:
        raise ValueError(f"Control-step/chunk mismatch in {path}")
    for row in rows:
        actions = row["action_chunk_raw23"]
        if len(actions) != 16 or any(len(action) != 23 for action in actions):
            raise ValueError(f"Invalid action chunk in {path}: {row['chunk_id']}")
        if row["high_level"]["context_id"] != row["low_level"]["context_id"]:
            raise ValueError(f"High/low context mismatch in {path}: {row['chunk_id']}")
    return rows


def wrapped_lines(row):
    high = row["high_level"]
    low = row["low_level"]
    status = "NEW HIGH-LEVEL INFERENCE" if row["high_level_called"] else "REUSED HIGH-LEVEL CONTEXT"
    sections = [
        f"Task: {row['task']}",
        f"Instance: {row['instance_id']}    env={row['env']}    chunk={row['chunk_id']}",
        f"Control step: {row['control_step']}    request={row['inference_request']}",
        f"{status}: {high['high_call_id']}",
        "",
        "[HIGH-LEVEL INPUT PROMPT]",
        high["prompt_text"],
        "",
        "[HIGH-LEVEL RAW OUTPUT]",
        high["raw_output_text"],
        "",
        "[MEMORY AFTER COMMIT]",
        high["memory_after"],
        "",
        "[LOW-LEVEL PROMPT FOR THIS ACTION CHUNK]",
        low["prompt_text"],
    ]
    lines = []
    for section in sections:
        for source_line in str(section).splitlines() or [""]:
            lines.extend(textwrap.wrap(source_line, width=92, replace_whitespace=False,
                                       drop_whitespace=False) or [""])
    return lines


def pages(row):
    lines = wrapped_lines(row)
    return [lines[index:index + LINES_PER_PAGE]
            for index in range(0, len(lines), LINES_PER_PAGE)] or [[""]]


def fit(image, size):
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    return result


def canvas_for(*, rollout_frame, keyframe, row, page_index, page_count):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE) if FONT_PATH.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(FONT_PATH), 14) if FONT_PATH.exists() else ImageFont.load_default()

    if rollout_frame is not None:
        top = fit(rollout_frame, (LEFT_WIDTH, 620))
        canvas.paste(top, ((LEFT_WIDTH - top.width) // 2, (620 - top.height) // 2))
        draw.text((18, 18), "OFFICIAL ROLLOUT VIDEO", fill=(255, 220, 80), font=small,
                  stroke_width=2, stroke_fill="black")
    exact = fit(keyframe, (LEFT_WIDTH - 20, 430 if rollout_frame is not None else HEIGHT - 20))
    exact_y = 635 + (435 - exact.height) // 2 if rollout_frame is not None else (HEIGHT - exact.height) // 2
    canvas.paste(exact, ((LEFT_WIDTH - exact.width) // 2, exact_y))
    draw.text((18, 650 if rollout_frame is not None else 18),
              "EXACT SERVER OBSERVATION AT CHUNK BOUNDARY", fill=(100, 230, 255), font=small,
              stroke_width=2, stroke_fill="black")

    draw.rectangle((LEFT_WIDTH, 0, WIDTH, HEIGHT), fill=(20, 23, 29))
    page_lines = pages(row)[page_index]
    title = f"ALIGNMENT TRACE    page {page_index + 1}/{page_count}"
    draw.text((LEFT_WIDTH + PANEL_MARGIN, PANEL_MARGIN), title, fill=(80, 220, 255), font=font)
    y = PANEL_MARGIN + LINE_HEIGHT * 2
    for line in page_lines:
        color = (245, 245, 245)
        if line.startswith("["):
            color = (255, 210, 90)
        elif "HIGH-LEVEL" in line:
            color = (120, 240, 170)
        draw.text((LEFT_WIDTH + PANEL_MARGIN, y), line, fill=color, font=font)
        y += LINE_HEIGHT
    return canvas


def open_output(path, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=rate, options={"preset": "fast", "crf": "23"})
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    return container, stream


def write_frame(container, stream, image):
    frame = av.VideoFrame.from_image(image)
    for packet in stream.encode(frame):
        container.mux(packet)


def close_output(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def render_realtime(source, destination, rows, trace_root):
    starts = [row["control_step"] for row in rows]
    source_container = av.open(str(source))
    source_stream = source_container.streams.video[0]
    rate = source_stream.average_rate or 30
    output, output_stream = open_output(destination, rate)
    frame_count = 0
    current_chunk = None
    keyframe = None
    try:
        for frame_count, frame in enumerate(source_container.decode(source_stream), start=1):
            index = frame_count - 1
            row_index = bisect_right(starts, index) - 1
            if row_index < 0:
                raise ValueError(f"Video begins before first trace boundary: {source}")
            row = rows[min(row_index, len(rows) - 1)]
            if current_chunk != row["chunk_id"]:
                current_chunk = row["chunk_id"]
                keyframe = Image.open(trace_root / row["observation"]["montage"]).convert("RGB")
            row_pages = pages(row)
            next_start = starts[row_index + 1] if row_index + 1 < len(starts) else max(index + 1, starts[row_index] + 16)
            span = max(1, next_start - starts[row_index])
            page_index = min(len(row_pages) - 1,
                             (index - starts[row_index]) * len(row_pages) // span)
            image = canvas_for(rollout_frame=frame.to_image(), keyframe=keyframe, row=row,
                               page_index=page_index, page_count=len(row_pages))
            write_frame(output, output_stream, image)
    finally:
        source_container.close()
        close_output(output, output_stream)
    return frame_count


def render_chunk_review(destination, rows, trace_root):
    output, stream = open_output(destination, 2)
    frames = 0
    try:
        for row in rows:
            keyframe = Image.open(trace_root / row["observation"]["montage"]).convert("RGB")
            row_pages = pages(row)
            for page_index in range(len(row_pages)):
                image = canvas_for(rollout_frame=None, keyframe=keyframe, row=row,
                                   page_index=page_index, page_count=len(row_pages))
                # Two frames at 2 fps: one readable second per text page.
                for _ in range(2):
                    write_frame(output, stream, image)
                    frames += 1
    finally:
        close_output(output, stream)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    task = args.task_output.name
    result = {"task": task, "instances": {}}
    for trace_file in sorted((args.trace_root / task).glob("*/chunks.jsonl")):
        instance = int(trace_file.parent.name)
        rows = load_rows(trace_file)
        source = args.task_output / "videos" / f"{task}_{instance}_0.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        realtime = args.output_dir / "realtime" / f"{task}_{instance}_0_text.mp4"
        review = args.output_dir / "chunk_review" / f"{task}_{instance}_0_chunks.mp4"
        source_frames = render_realtime(source, realtime, rows, args.trace_root)
        review_frames = render_chunk_review(review, rows, args.trace_root)
        if source_frames > rows[-1]["control_step"] + 16:
            raise ValueError(f"Trace ends before rollout video for instance {instance}")
        result["instances"][str(instance)] = {
            "chunks": len(rows),
            "source_video": str(source),
            "source_frames": source_frames,
            "realtime_text_video": str(realtime),
            "chunk_review_video": str(review),
            "chunk_review_frames": review_frames,
            "trace_jsonl": str(trace_file),
        }
    if set(result["instances"]) != {"301", "302", "303"}:
        raise ValueError(f"Expected trace for instances 301/302/303, got {sorted(result['instances'])}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alignment_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
