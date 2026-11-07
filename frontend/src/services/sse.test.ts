import { SseParser } from "./sse";

test("parses several events in one chunk", () => {
  const p = new SseParser();
  const out = p.push('event: token\ndata: {"a":1}\n\nevent: done\ndata: {}\n\n');
  expect(out).toEqual([
    { event: "token", data: '{"a":1}' },
    { event: "done", data: "{}" },
  ]);
});

test("handles events split across arbitrary chunk boundaries", () => {
  const p = new SseParser();
  const text = 'event: token\ndata: {"text": "x"}\n\nevent: done\ndata: {}\n\n';
  const out = [];
  for (const ch of text) out.push(...p.push(ch)); // one character at a time
  expect(out.map((e) => e.event)).toEqual(["token", "done"]);
  expect(out[0]?.data).toBe('{"text": "x"}');
});

test("normalizes CRLF, including a split CRLF, and ignores comments", () => {
  const p = new SseParser();
  expect(p.push(": keepalive\r")).toEqual([]);
  expect(p.push("\nevent: a\r\ndata: 1\r\n\r\n")).toEqual([{ event: "a", data: "1" }]);
});

test("an incomplete trailing event is not emitted", () => {
  expect(new SseParser().push("event: token\ndata: {}")).toEqual([]);
});