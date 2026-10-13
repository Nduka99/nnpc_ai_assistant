import { expect, test } from "vitest";
import { hasOwnMark, pageOf, quoted, readableDate } from "./format";

test("dates read as day, short month and year, and odd values pass through", () => {
  expect(readableDate("2026-09-29")).toBe("29 Sep 2026");
  expect(readableDate("2025-02-05")).toBe("5 Feb 2025");
  expect(readableDate("")).toBe("");
  expect(readableDate("September 2026")).toBe("September 2026");
});

test("a PDF link gives its page, any other link none", () => {
  expect(pageOf("https://cms1977.nnpcgroup.com/uploads/report.pdf#page=12")).toBe(12);
  expect(pageOf("https://nnpcgroup.com/netco#:~:text=NETCO")).toBeNull();
});

test("a quote is set in quotation marks once, whatever it brings", () => {
  expect(hasOwnMark("“Revenue rose,” it said.")).toBe(true);
  expect(hasOwnMark("Revenue rose.")).toBe(false);
  expect(quoted("Revenue rose.")).toBe("“Revenue rose.”");
  expect(quoted("“Revenue rose,” it said.")).toBe("“Revenue rose,” it said.");
});
