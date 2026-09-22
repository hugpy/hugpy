import { useState } from "react";
import type { MediaInputValue } from "./../types";
import { extractUploadedFiles } from "./UploadUtils";
import {FILE_UPLOAD_ENDPOINT} from './../constants';
import { request, describeAppError, errorOf } from "./../../../../transport/client";
import { getSessionId } from "./../../../../session";
interface UseMediaInputReceptacleArgs {
  value: MediaInputValue;
  onChange: (next: MediaInputValue) => void;
}

export function useMediaInputReceptacle({
  value,
  onChange,
}: UseMediaInputReceptacleArgs) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [lastUploadResponse, setLastUploadResponse] = useState<any>(null);

  function patch(next: Partial<MediaInputValue>) {
    onChange({ ...value, ...next });
  }

  async function uploadFiles(files: File[]) {
    setUploadError("");

    if (!files.length) {
      patch({
        files: [],
        uploadedFiles: [],
        selectedIds: [],
      });
      return;
    }

    const formData = new FormData();

    for (const file of files) {
      // Field name MUST be "file" to match the /uploads route
      // (request.files.get("file")) and the deployed bundle. Prior "files"
      // drifted from the backend and silently broke uploads on rebuild.
      formData.append("file", file);
    }
    formData.append("sid", getSessionId()); // tag uploads to this session for per-session wipe

    setUploading(true);

    try {
      const res = await request(FILE_UPLOAD_ENDPOINT, {
        method: "POST",
        body: formData,
        meta: { specKey: "upload" },
      });

      if (!res.ok) {
        setLastUploadResponse(null);
        throw new Error(describeAppError(errorOf(res)));
      }

      const data = res.value;
      setLastUploadResponse(data);

      const uploadedFiles = extractUploadedFiles(data);

      patch({
        files,
        uploadedFiles,
        selectedIds: uploadedFiles.map((file) => file.id),
      });
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));

      patch({
        files,
        uploadedFiles: [],
        selectedIds: [],
      });
    } finally {
      setUploading(false);
    }
  }

  function setFiles(files: FileList | File[]) {
    uploadFiles(Array.from(files));
  }

  function toggleSelected(id: string) {
    const nextSelected = new Set(value.selectedIds);

    if (nextSelected.has(id)) {
      nextSelected.delete(id);
    } else {
      nextSelected.add(id);
    }

    patch({ selectedIds: Array.from(nextSelected) });
  }

  function selectAll() {
    patch({
      selectedIds: value.uploadedFiles.map((file) => file.id),
    });
  }

  function clearSelection() {
    patch({ selectedIds: [] });
  }

  const allSelected =
    value.uploadedFiles.length > 0 &&
    value.selectedIds.length === value.uploadedFiles.length;

  return {
    dragActive,
    setDragActive,
    uploading,
    uploadError,
    lastUploadResponse,
    patch,
    setFiles,
    toggleSelected,
    selectAll,
    clearSelection,
    allSelected,
  };
}
