import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from yolov8 import CLASSES, post_process

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise ImportError('Please install rknn-toolkit-lite2 on RK3588 device.') from exc


IMG_SIZE = (640, 640)  # width, height


@dataclass
class FrameTask:
    stream_id: int
    frame_id: int
    frame_bgr: np.ndarray


@dataclass
class FrameResult:
    stream_id: int
    frame_id: int
    infer_ms: float
    boxes: Optional[np.ndarray]
    classes: Optional[np.ndarray]
    scores: Optional[np.ndarray]
    frame_bgr: np.ndarray


class VideoStreamReader(threading.Thread):
    def __init__(self, stream_id: int, source: str, frame_q: queue.Queue):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.frame_q = frame_q
        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f'stream-{stream_id} open failed: {source}')

    def run(self):
        frame_id = 0
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            item = FrameTask(stream_id=self.stream_id, frame_id=frame_id, frame_bgr=frame)
            frame_id += 1

            # keep latest frame (drop old) to avoid long queue latency
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                    self.frame_q.task_done()
                except queue.Empty:
                    pass
            self.frame_q.put(item)

        self.cap.release()

    def stop(self):
        self.stop_event.set()


def letterbox_with_info(image: np.ndarray, dst_size=(640, 640), color=(0, 0, 0)):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = dst_size

    r = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * r)), int(round(src_h * r))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad = np.full((dst_h, dst_w, 3), color, dtype=np.uint8)

    dw = (dst_w - new_w) / 2.0
    dh = (dst_h - new_h) / 2.0
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    pad[top:top + new_h, left:left + new_w] = resized
    return pad, r, dw, dh


def restore_boxes_to_src(boxes: np.ndarray, src_shape, r: float, dw: float, dh: float):
    out = boxes.copy()
    h, w = src_shape[:2]
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / r
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def draw_detections(frame: np.ndarray, boxes, scores, classes):
    if boxes is None:
        return frame
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = f'{CLASSES[int(cl)].strip()} {float(score):.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


class RKNNWorker:
    def __init__(self, model_path: str, core_mask: Optional[int], worker_name: str):
        self.worker_name = worker_name
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'[{worker_name}] load_rknn failed: {ret}')

        ret = self.rknn.init_runtime() if core_mask is None else self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f'[{worker_name}] init_runtime failed: {ret}')

    def infer(self, frame_bgr: np.ndarray):
        inp, r, dw, dh = letterbox_with_info(frame_bgr, IMG_SIZE)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = self.rknn.inference(inputs=[inp])
        infer_ms = (time.perf_counter() - t0) * 1000.0

        boxes, classes, scores = post_process(outputs)
        if boxes is not None:
            boxes = restore_boxes_to_src(boxes, frame_bgr.shape, r, dw, dh)
        return infer_ms, boxes, classes, scores

    def release(self):
        self.rknn.release()


def build_core_mask(worker_idx: int):
    masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    return masks[worker_idx % 3]


def worker_loop(ctx: RKNNWorker, task_q: queue.Queue, result_q: queue.Queue):
    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        infer_ms, boxes, classes, scores = ctx.infer(task.frame_bgr)
        result_q.put(
            FrameResult(
                stream_id=task.stream_id,
                frame_id=task.frame_id,
                infer_ms=infer_ms,
                boxes=boxes,
                classes=classes,
                scores=scores,
                frame_bgr=task.frame_bgr,
            )
        )
        task_q.task_done()


def parse_video_sources(arg: str) -> List[str]:
    if os.path.isdir(arg):
        all_files = sorted(os.listdir(arg))
        return [os.path.join(arg, f) for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return [v.strip() for v in arg.split(',') if v.strip()]


def run(args):
    sources = parse_video_sources(args.video_sources)
    if not sources:
        raise ValueError('No valid video source found.')

    if args.streams > len(sources):
        print(f'[WARN] streams={args.streams} > sources={len(sources)}, auto clamp.')
        args.streams = len(sources)
    sources = sources[:args.streams]

    stream_queues = [queue.Queue(maxsize=1) for _ in range(args.streams)]
    readers = [VideoStreamReader(i, src, stream_queues[i]) for i, src in enumerate(sources)]
    for r in readers:
        r.start()

    task_q = queue.Queue(maxsize=args.workers * 3)
    result_q = queue.Queue(maxsize=args.workers * 3)

    workers = []
    worker_threads = []
    for wid in range(args.workers):
        mask = build_core_mask(wid) if args.bind_cores else None
        w = RKNNWorker(args.model_path, mask, worker_name=f'worker-{wid}')
        t = threading.Thread(target=worker_loop, args=(w, task_q, result_q), daemon=True)
        t.start()
        workers.append(w)
        worker_threads.append(t)

    writers = {}
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    stream_done = [False] * args.streams
    stream_processed = [0] * args.streams
    frame_budget = args.frames_per_stream

    t_start = time.perf_counter()
    while True:
        if all(stream_done):
            break

        # Round-robin feed
        for sid in range(args.streams):
            if stream_done[sid]:
                continue
            if frame_budget > 0 and stream_processed[sid] >= frame_budget:
                stream_done[sid] = True
                continue

            try:
                task = stream_queues[sid].get_nowait()
                task_q.put(task, timeout=0.05)
            except queue.Empty:
                if not readers[sid].is_alive() and stream_queues[sid].empty():
                    stream_done[sid] = True
            except queue.Full:
                pass

        # consume result and draw
        try:
            res = result_q.get(timeout=0.02)
            stream_processed[res.stream_id] += 1

            vis = draw_detections(res.frame_bgr, res.boxes, res.scores, res.classes)
            if args.save_dir:
                if res.stream_id not in writers:
                    h, w = vis.shape[:2]
                    out_path = os.path.join(args.save_dir, f'stream_{res.stream_id}.mp4')
                    writers[res.stream_id] = cv2.VideoWriter(
                        out_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        args.output_fps,
                        (w, h),
                    )
                writers[res.stream_id].write(vis)

            if args.show:
                cv2.imshow(f'stream_{res.stream_id}', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stream_done = [True] * args.streams

        except queue.Empty:
            pass

    for rd in readers:
        rd.stop()
    for rd in readers:
        rd.join(timeout=1.0)

    for _ in range(args.workers):
        task_q.put(None)
    task_q.join()
    for th in worker_threads:
        th.join()
    for w in workers:
        w.release()

    for wr in writers.values():
        wr.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    total_frames = sum(stream_processed)
    fps = total_frames / elapsed if elapsed > 0 else 0.0
    print('\n===== Multi-stream video summary =====')
    print(f'sources            : {len(sources)}')
    print(f'workers            : {args.workers}')
    print(f'bind_cores         : {args.bind_cores}')
    print(f'processed frames   : {total_frames}')
    print(f'elapsed(s)         : {elapsed:.3f}')
    print(f'overall FPS        : {fps:.2f}')
    print(f'per-stream frames  : {stream_processed}')


def get_args():
    parser = argparse.ArgumentParser(description='RK3588 YOLOv8 multi-stream VIDEO inference with draw boxes.')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN model path')
    parser.add_argument('--video_sources', type=str, required=True, help='Comma paths or a folder of videos')
    parser.add_argument('--streams', type=int, default=9, help='logical stream count')
    parser.add_argument('--frames_per_stream', type=int, default=300, help='<=0 means run until video ends')
    parser.add_argument('--workers', type=int, default=3, help='NPU worker count, 3 for RK3588')
    parser.add_argument('--bind_cores', action='store_true', help='bind worker 0/1/2 to core 0/1/2')
    parser.add_argument('--save_dir', type=str, default='./result_video', help='output directory for visualized videos')
    parser.add_argument('--output_fps', type=float, default=25.0, help='output video fps')
    parser.add_argument('--show', action='store_true', help='display windows during running')
    return parser.parse_args()


if __name__ == '__main__':
    run(get_args())
