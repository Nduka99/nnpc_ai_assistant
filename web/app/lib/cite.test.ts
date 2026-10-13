import { expect, test } from "vitest";
import { citationText } from "./cite";
import { view } from "./fixtures";

test("an answer copies as claims with source numbers, then dated sources", () => {
  const answer = view({
    claims: [
      { text: "NETCO visits secondary schools.", sources: [1] },
      { text: "Its 2025 report says so.", sources: [1, 2] },
    ],
    sources: [
      ...view().sources,
      {
        number: 2,
        title: "NETCO Annual Report",
        heading: "",
        quote: "“Schools outreach continued in 2025,” it said.",
        link: "https://nnpcgroup.com/report.pdf#page=4",
        url: "https://nnpcgroup.com/report.pdf",
        published: "2026-03-02",
        captured: "2026-10-10",
        compiled: false,
      },
    ],
  });

  expect(citationText(answer)).toBe(
    [
      "NETCO visits secondary schools. [1]",
      "Its 2025 report says so. [1][2]",
      "",
      "Sources",
      "[1] “NETCO visits secondary schools.” NNPC Engineering & Technical Company Limited (NETCO) › Community, retrieved 10 Oct 2026. https://nnpcgroup.com/netco#:~:text=NETCO%20visits",
      "[2] “Schools outreach continued in 2025,” it said. NETCO Annual Report, p. 4, published 2 Mar 2026, retrieved 10 Oct 2026. https://nnpcgroup.com/report.pdf#page=4",
      "",
      "From PetroKnowledge, independent of NNPC Limited.",
    ].join("\n"),
  );
});
