import { useCallback, useEffect, useRef, useState } from "react";

import { isAbortError, streamGenerate } from "../services/stream";
import type { GenerateRequest, StreamDone, StreamStart, StreamToken } from "../types/api";

export type StreamStatus = "idle" | "streaming" | "done" | "stopped" | "error";

export interface StreamState {
  status: StreamStatus;
  request: GenerateRequest | null;
  tokens: StreamToken[];
  start: StreamStart | null;
  done: StreamDone | null;
  error: string | null;
}

export interface FinishedRun {
  request: GenerateRequest;
  completion: string;
  status: "done" | "stopped";
  tokens: number;
  seconds: number | null;
  tokensPerSecond: number | null;
  device: string | null;
}

const IDLE: StreamState = {
  status: "idle", request: null, tokens: [], start: null, done: null, error: null,
};

export function useGenerationStream() {
  const [state, setState] = useState<StreamState>(IDLE);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => abortRef.current?.abort(), []);
  useEffect(() => () => abortRef.current?.abort(), []); // abort on unmount

  const start = useCallback(
    async (request: GenerateRequest, onFinish?: (run: FinishedRun) => void) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const tokens: StreamToken[] = [];
      let startEvent: StreamStart | null = null;
      let doneEvent: StreamDone | null = null;
      setState({ ...IDLE, status: "streaming", request });

      const finish = (status: "done" | "stopped") =>
        onFinish?.({
          request,
          completion: tokens.map((t) => t.text).join(""),
          status,
          tokens: tokens.length,
          seconds: doneEvent?.seconds ?? null,
          tokensPerSecond: doneEvent?.tokens_per_second ?? null,
          device: startEvent?.device ?? null,
        });

      try {
        await streamGenerate(
          request,
          {
            onStart: (e) => {
              startEvent = e;
              setState((s) => ({ ...s, start: e }));
            },
            onToken: (e) => {
              tokens.push(e);
              setState((s) => ({ ...s, tokens: [...tokens] }));
            },
            onDone: (e) => {
              doneEvent = e;
              setState((s) => ({ ...s, status: "done", done: e }));
            },
          },
          controller.signal,
        );
        finish("done");
      } catch (e) {
        if (isAbortError(e)) {
          setState((s) => ({ ...s, status: "stopped" }));
          finish("stopped");
        } else {
          setState((s) => ({
            ...s, status: "error", error: e instanceof Error ? e.message : "Generation failed",
          }));
        }
      }
    },
    [],
  );

  return { state, start, stop };
}