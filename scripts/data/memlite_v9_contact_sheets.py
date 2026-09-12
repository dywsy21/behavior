"""Extract original video frames for visual review; never synthesize images."""
from pathlib import Path
import argparse
import json

import av
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


CAMERAS = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"]


def decode_at(path, requested):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(requested / stream.time_base)), stream=stream, backward=True, any_frame=False)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            distance = abs(actual - requested)
            if best is None or distance < best[0]:
                best = distance, actual, frame.to_image()
            if actual >= requested:
                break
        if best is None or best[0] > 1.0 / 60 + 0.002:
            raise ValueError(f"Cannot align exact demo frame: {path} t={requested}, best={None if best is None else best[:2]}")
        return best[2], best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--dataset", default="/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4")
    args = parser.parse_args()
    review, dataset = Path(args.review), Path(args.dataset)
    samples = [json.loads(line) for line in (review / "manual_review_samples.jsonl").read_text().splitlines()]
    meta = {row["episode_index"]: row for file in sorted((dataset / "meta/episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(file).to_pylist()}
    records = []
    for task in range(5):
        candidates = [row for row in samples if row["task_index"] == task]
        chosen, used = [], set()
        for category in ("start", "ongoing", "transition", "corrupt_memory", "terminal", "parallel"):
            match = next((row for row in candidates if row["review_category"] == category
                          and (row["episode_index"], row["frame_index"]) not in used), None)
            if match:
                chosen.append(match)
                used.add((match["episode_index"], match["frame_index"]))
        for row in candidates:
            if len(chosen) >= 6:
                break
            key = row["episode_index"], row["frame_index"]
            if key not in used:
                chosen.append(row)
                used.add(key)
        assert len(chosen) == 6, (task, len(chosen))
        sheet = Image.new("RGB", (900, 6 * 326 + 30), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"Task {task} | original demo observations | head / left wrist / right wrist", fill="black")
        for index, row in enumerate(chosen):
            ep = meta[row["episode_index"]]
            y = 30 + index * 326
            draw.text((8, y + 2), f'ep={row["episode_index"]} frame={row["frame_index"]} category={row["review_category"]} status={row["intent_status"]}', fill="black")
            for camera_index, camera in enumerate(CAMERAS):
                prefix = "videos/" + camera
                video = dataset / f'videos/{camera}/chunk-{ep[prefix + "/chunk_index"]:03d}/file-{ep[prefix + "/file_index"]:03d}.mp4'
                requested = float(ep[prefix + "/from_timestamp"]) + row["frame_index"] / 30.0
                original, actual = decode_at(video, requested)
                filename = f'task{task}_ep{row["episode_index"]}_frame{row["frame_index"]}_cam{camera_index}.png'
                original.save(review / filename)
                thumbnail = original.copy()
                thumbnail.thumbnail((298, 298))
                sheet.paste(thumbnail, (camera_index * 300, y + 23))
                records.append(dict(task=task, episode=row["episode_index"], frame=row["frame_index"],
                                    category=row["review_category"], camera=camera, source=str(video), image=filename,
                                    requested_time=requested, decoded_time=actual, time_error_ms=1000 * abs(actual - requested)))
        sheet.save(review / f"task{task}_contact_sheet.png")
    (review / "visual_review_index.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"original_images": len(records), "unique_visual_moments": len(records) // 3,
                      "max_time_error_ms": max(r["time_error_ms"] for r in records), "human_review": "PENDING"}), flush=True)


if __name__ == "__main__":
    main()
