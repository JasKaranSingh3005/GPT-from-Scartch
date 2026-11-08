import { useEffect, useState } from "react";

import { API_BASE } from "../services/api";
import type { JobStatus, MetricRow } from "../types/api";

export type SocketState = "connecting" | "open" | "closed";

export interface TrainingSocket {
  rows: MetricRow[];
  state: SocketState;
  terminal: JobStatus | null;
  bestCheckpoint: { iter: number; val_loss: number } | null;
}

const WS_BASE = API_BASE.replace(/^http/, "ws").replace(/\/api\/v1$/, "");
const TERMINAL: Record<string, JobStatus> = {
  "training.completed": "COMPLETED", "training.failed": "FAILED", "training.stopped": "STOPPED",
};

/** Add or replace a row by iteration; keeps rows sorted. Replays are therefore harmless. */
export function mergeRow(rows: MetricRow[], row: MetricRow): MetricRow[] {
  const next = rows.filter((r) => r.iter !== row.iter);
  next.push(row);
  return next.sort((a, b) => a.iter - b.iter);
}

/** Live metrics for a job. Reconnects with backoff; every connect replays the history. */
export function useTrainingSocket(jobId: string | null): TrainingSocket {
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [state, setState] = useState<SocketState>("connecting");
  const [terminal, setTerminal] = useState<JobStatus | null>(null);
  const [best, setBest] = useState<TrainingSocket["bestCheckpoint"]>(null);

  useEffect(() => {
    setRows([]);
    setTerminal(null);
    setBest(null);
    if (jobId === null) {
      setState("closed");
      return;
    }
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    let finished = false;
    let disposed = false;

    const connect = () => {
      setState("connecting");
      ws = new WebSocket(`${WS_BASE}/ws/training/${jobId}`);
      ws.onopen = () => {
        attempt = 0;
        setState("open");
      };
      ws.onmessage = (msg: MessageEvent<string>) => {
        const { event, data } = JSON.parse(msg.data) as { event: string; data: Record<string, unknown> };
        if (event === "training.metrics") {
          const d = data as unknown as MetricRow;
          setRows((rs) => mergeRow(rs, {
            iter: d.iter, train_loss: d.train_loss, val_loss: d.val_loss, elapsed_s: d.elapsed_s,
          }));
        } else if (event === "training.checkpoint_saved") {
          setBest({ iter: data.iter as number, val_loss: data.val_loss as number });
        } else if (event in TERMINAL) {
          finished = true;
          setTerminal(TERMINAL[event] ?? null);
        } else if (event === "error") {
          finished = true; // e.g. unknown job: do not retry
        }
      };
      ws.onclose = () => {
        setState("closed");
        if (finished || disposed) return;
        attempt += 1;
        timer = setTimeout(connect, Math.min(10000, 500 * 2 ** attempt));
      };
    };
    connect();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    };
  }, [jobId]);

  return { rows, state, terminal, bestCheckpoint: best };
}