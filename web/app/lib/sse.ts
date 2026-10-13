/** One server-sent event: its name (`event:`) and its data lines joined. */
export interface ServerEvent {
  event: string;
  data: string;
}

/**
 * A parser for a server-sent event stream that arrives in chunks of any size:
 * call the returned function with each chunk, and `onEvent` hears each complete
 * event (WHATWG HTML, "Server-sent events"). Comment lines, such as keep-alives,
 * are skipped, and an event without data is not dispatched.
 */
export function createParser(onEvent: (event: ServerEvent) => void) {
  let buffer = "";
  return (chunk: string): void => {
    // Joined before normalising, so a CRLF split between chunks stays one break.
    buffer = (buffer + chunk).replace(/\r\n/g, "\n");
    let end = buffer.indexOf("\n\n");
    while (end !== -1) {
      dispatch(buffer.slice(0, end), onEvent);
      buffer = buffer.slice(end + 2);
      end = buffer.indexOf("\n\n");
    }
  };
}

function dispatch(block: string, onEvent: (event: ServerEvent) => void): void {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  if (data.length > 0) onEvent({ event, data: data.join("\n") });
}
