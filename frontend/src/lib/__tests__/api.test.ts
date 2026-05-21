import { describe, expect, it } from "vitest";
import { parseSSEFrame } from "../api";

describe("parseSSEFrame", () => {
  it("parses a token frame (plain text data)", () => {
    const raw = "event: token\ndata: Olá, como";
    const frame = parseSSEFrame(raw);
    expect(frame).toEqual({ event: "token", data: "Olá, como" });
  });

  it("joins multi-line data fields with newline", () => {
    const raw = "event: token\ndata: linha 1\ndata: linha 2";
    const frame = parseSSEFrame(raw);
    expect(frame).toEqual({ event: "token", data: "linha 1\nlinha 2" });
  });

  it("parses a JSON-carrying event", () => {
    const payload = JSON.stringify({ session_id: "s1" });
    const raw = `event: done\ndata: ${payload}`;
    const frame = parseSSEFrame(raw);
    expect(frame?.event).toBe("done");
    expect(JSON.parse(frame!.data)).toEqual({ session_id: "s1" });
  });

  it("returns null for empty frame", () => {
    expect(parseSSEFrame("")).toBeNull();
    expect(parseSSEFrame("event: done")).toBeNull();
  });
});
