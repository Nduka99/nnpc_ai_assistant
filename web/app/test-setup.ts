import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/vue";
import { afterEach } from "vitest";

// jsdom lays nothing out: it has no ResizeObserver and no scrolling. The stand-ins
// report each observed element once, at its (zero) size, and scroll nowhere.
class ResizeObserverStandIn {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe(target: Element): void {
    const entry = { target } as ResizeObserverEntry;
    this.callback([entry], this as unknown as ResizeObserver);
  }
  unobserve(): void {}
  disconnect(): void {}
}
globalThis.ResizeObserver = ResizeObserverStandIn as unknown as typeof ResizeObserver;
Element.prototype.scrollIntoView = () => {};

// Testing Library unmounts after each test only where test globals exist; here they
// are imported, so it is done explicitly. The saved thread is cleared too.
afterEach(() => {
  cleanup();
  sessionStorage.clear();
});
