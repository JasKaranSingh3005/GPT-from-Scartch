export interface SseEvent {
  event: string;
  data: string;
}

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // blank or comment
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length === 0 ? null : { event, data: data.join("\n") };
}

/** Incremental Server-Sent Events parser: feed it text chunks of any size. */
export class SseParser {
  private buffer = "";

  push(chunk: string): SseEvent[] {
    // Normalizing the whole buffer also handles a "\r\n" split across two chunks.
    this.buffer = (this.buffer + chunk).replace(/\r\n/g, "\n");
    const events: SseEvent[] = [];
    let end = this.buffer.indexOf("\n\n");
    while (end !== -1) {
      const parsed = parseBlock(this.buffer.slice(0, end));
      this.buffer = this.buffer.slice(end + 2);
      if (parsed) events.push(parsed);
      end = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}