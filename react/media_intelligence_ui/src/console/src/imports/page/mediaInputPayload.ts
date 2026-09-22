import type { MediaInputValue, UploadedFileRef } from "./../pages/pageSpec";

export type ResolvedMediaInput = {
  inputMode: MediaInputValue["inputMode"];
  text?: string;
  url?: string;

  /**
   * Server-side uploaded file ids (opaque handles).
   * This is usually what HugPy should consume instead of a browser File object.
   */
  fileIds: string[];

  /**
   * Full uploaded file objects, if the tool needs names/types/sizes.
   */
  uploadedFiles: UploadedFileRef[];

  /**
   * Local browser File objects. Useful only if the endpoint expects multipart upload
   * at execution time instead of using already-uploaded paths.
   */
  localFiles: File[];
};

export function resolveMediaInput(input: MediaInputValue): ResolvedMediaInput {
  const uploadedFiles = input.uploadedFiles ?? [];
  const selectedIds = input.selectedIds ?? [];

  const selectedUploadedFiles =
    selectedIds.length > 0
      ? uploadedFiles.filter((file) => selectedIds.includes(file.id))
      : uploadedFiles;

  return {
    inputMode: input.inputMode,
    text: input.text,
    url: input.url,
    fileIds: selectedUploadedFiles.map((file) => file.id),
    uploadedFiles: selectedUploadedFiles,
    localFiles: input.files ?? [],
  };
}
