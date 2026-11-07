import type { GenerateRequest, StreamDone, StreamStart, StreamToken } from "../types/api";
import { API_BASE, ApiError, errorFromResponse } from "./api";
import { SseParser } from "./sse";

export interface StreamHandlers {
  onStart: (e: StreamStart) => void;
  onToken: (e: StreamToken) => void;
  onDone: (e: StreamDone) => void;
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

function parseJson(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    throw new ApiError(0, "The server sent a malformed stream event");
  }
}

/**
 * POST /generate/stream and dispatch events as they arrive. Rejects with ApiError on
 * HTTP errors, a server-reported `error` event, or a stream that ends without `done`.
 * Aborting through `signal` rejects with an AbortError (see isAbortError).
 */
export async function streamGenerate(
  body: GenerateRequest,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError(0, `Cannot reach the backend at ${API_BASE}`);
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError(0, "This browser cannot read streaming responses");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  let finished = false;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) {
      const payload = parseJson(ev.data);
      if (ev.event === "start") handlers.onStart(payload as StreamStart);
      else if (ev.event === "token") handlers.onToken(payload as StreamToken);
      else if (ev.event === "done") {
        finished = true;
        handlers.onDone(payload as StreamDone);
      } else if (ev.event === "error") {
        const detail = (payload as { detail?: unknown }).detail;
        throw new ApiError(500, typeof detail === "string" ? detail : "Generation failed");
      }
    }
  }
  if (!finished) throw new ApiError(0, "The stream ended before generation finished");
}