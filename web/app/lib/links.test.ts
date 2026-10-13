import { expect, test } from "vitest";
import { safeLink } from "./links";

test("links to the sources' hosts are kept", () => {
  for (const address of [
    "https://nnpcgroup.com/businesses#:~:text=NETCO",
    "https://cms1977.nnpcgroup.com/uploads/2024_Annual_Report.pdf#page=2",
    "https://en.wikipedia.org/wiki/NNPC",
    "https://www.nuprc.gov.ng/laws/regulations/gazetted",
    "http://www.eia.gov/international/analysis/country/NGA",
  ]) {
    expect(safeLink(address)).toBe(new URL(address).href);
  }
});

test("other hosts, look-alikes, scripts and malformed addresses are not links", () => {
  for (const address of [
    "https://example.com/nnpc",
    "https://evil-nnpcgroup.com/businesses",
    "https://nnpcgroup.com.example.com/businesses",
    "https://de.wikipedia.org/wiki/NNPC",
    "javascript:alert(1)",
    "ftp://nnpcgroup.com/file",
    "not a url",
  ]) {
    expect(safeLink(address)).toBeNull();
  }
});
