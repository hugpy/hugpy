import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import { FileInputStage } from "./FileInputStage";
import type { MediaInputValue } from "./../../imports";

const value: MediaInputValue = {
  inputMode: "file",
  text: "",
  url: "",
  files: [],
  uploadedFiles: [{ name: "a.png", id: "id-a", type: "image/png", size: 2048 }],
  selectedIds: [],
};

function renderStage() {
  return render(
    <FileInputStage
      value={value}
      dragActive={false}
      uploading={false}
      uploadError=""
      allSelected={false}
      onSetDragActive={() => {}}
      onSetFiles={() => {}}
      onToggleSelected={() => {}}
      onSelectAll={() => {}}
      onClearSelection={() => {}}
    />,
  );
}

describe("FileInputStage accessibility (Phase 11)", () => {
  it("drop zone is a keyboard-operable button", () => {
    const { getByLabelText } = renderStage();
    const zone = getByLabelText(/upload files/i);
    expect(zone.getAttribute("role")).toBe("button");
    expect(zone.getAttribute("tabindex")).toBe("0");
  });

  it("has no axe violations", async () => {
    const { container } = renderStage();
    const results = await axe(container);
    expect(results.violations.map((v) => v.id)).toEqual([]);
  });
});
