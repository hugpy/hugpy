/*
 * Composer.tsx — the unified-composer pill.
 *
 * Visual rules from convo (this is what changed from the previous version):
 *
 *   - The composer is ONE SURFACE. No nested grid for sampling controls.
 *     Sampling controls render OUTSIDE the composer, in a thin strip
 *     below it — managed by main.tsx. This keeps the pill shape clean.
 *
 *   - Grid is simpler now: leading | primary | trailing. No footer row.
 *
 *   - Radius is --composer-radius (32px). Background is
 *     --bg-elevated-primary. Subtle 1px border + soft shadow.
 *
 *   - Send button is small (32px) and round. Disabled at rest;
 *     fills to --text-primary the moment the textarea has content.
 *
 *   - Plus button on the left, mic + send on the right.
 *
 *   - Textarea auto-grows up to 30svh, then internal scroll.
 */

import { useEffect, useRef, useState } from "react";
import type { ClipboardEvent, FormEvent, KeyboardEvent } from "react";
import { Icon } from "./Icons";

// Minimal slice of the Web Speech API we rely on for dictation. The standard TS
// DOM lib doesn't ship these types (and webkitSpeechRecognition never will), so
// we declare exactly what we touch.
interface MinimalRecognitionEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}
interface MinimalRecognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: MinimalRecognitionEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}
type RecognitionCtor = new () => MinimalRecognition;

interface ComposerProps {
  prompt: string;
  loading: boolean;
  modelLabel: string;
  onPromptChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onStop: () => void;
  onAttachFiles?: (files: File[]) => void;
  uploading?: boolean;
  /** A bare file attachment (no typed text) is a valid submit. */
  hasAttachment?: boolean;
}

export default function Composer({
  prompt,
  loading,
  modelLabel,
  onPromptChange,
  onSubmit,
  onStop,
  onAttachFiles,
  uploading,
  hasAttachment,
}: ComposerProps): JSX.Element {
  // A bare attachment (no typed text) is a valid submit — mirror submitPrompt so
  // the visible Send button matches the rule the form actually enforces.
  const canSubmit = (Boolean(prompt.trim()) || Boolean(hasAttachment)) && !loading;
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // ── dictation (Web Speech API) ───────────────────────────────────────
  // Browser-native speech-to-text; no backend. Appends to the current draft.
  // Gracefully disabled where the API is absent (e.g. Firefox).
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<MinimalRecognition | null>(null);
  const dictationBaseRef = useRef("");
  const promptRef = useRef(prompt);
  useEffect(() => {
    promptRef.current = prompt;
  }, [prompt]);

  const RecognitionImpl: RecognitionCtor | undefined =
    typeof window !== "undefined"
      ? (
          window as unknown as {
            SpeechRecognition?: RecognitionCtor;
            webkitSpeechRecognition?: RecognitionCtor;
          }
        ).SpeechRecognition ??
        (
          window as unknown as {
            webkitSpeechRecognition?: RecognitionCtor;
          }
        ).webkitSpeechRecognition
      : undefined;
  const dictationSupported = Boolean(RecognitionImpl);

  // If generation takes over, end any in-flight dictation.
  useEffect(() => {
    if (loading) recognitionRef.current?.stop();
  }, [loading]);

  function toggleDictation() {
    if (!RecognitionImpl) return;
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const recognition = new RecognitionImpl();
    recognition.lang =
      (typeof navigator !== "undefined" && navigator.language) || "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    // Seed from the current draft so speech appends rather than overwrites it.
    dictationBaseRef.current = promptRef.current.trim();
    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i += 1) {
        transcript += event.results[i][0]?.transcript ?? "";
      }
      const base = dictationBaseRef.current;
      const joiner = base && transcript ? " " : "";
      onPromptChange(base + joiner + transcript.trimStart());
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
    };
    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  // Auto-grow the textarea up to its max-height (CSS handles the cap).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [prompt]);

  function handlePromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    if (loading) return;
    event.currentTarget.form?.requestSubmit();
  }

  // Pasted files (screenshots, copied files) attach like a drop; plain-text
  // paste falls through to the textarea unchanged. Mirrors the form's onDrop.
  function handlePaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    if (!onAttachFiles) return;
    const files = Array.from(event.clipboardData.files);
    if (files.length) {
      event.preventDefault();
      onAttachFiles(files);
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      onDragOver={(e) => {
        if (onAttachFiles) e.preventDefault();
      }}
      onDrop={(e) => {
        if (!onAttachFiles) return;
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files);
        if (files.length) onAttachFiles(files);
      }}
      className="group/composer w-full"
      data-type="unified-composer"
    >
      <div
        data-composer-surface="true"
        className="
          corner-superellipse/1.1
          relative grid items-end gap-2
          grid-cols-[auto_1fr_auto]
          [grid-template-areas:'leading_primary_trailing']
          p-2
          border border-token-border-light
          shadow-short-composer
          transition-shadow duration-150
          focus-within:shadow-lg
        "
        style={{ background: "var(--bg-elevated-primary)" }}
      >
        {/* leading ------------------------------------------------ */}
        <div className="[grid-area:leading] pb-0.5">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              if (files.length) onAttachFiles?.(files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            id="composer-plus-btn"
            data-testid="composer-plus-btn"
            aria-label="Attach a file"
            title="Attach a file"
            className="composer-btn"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            <Icon name="plus" width={17} height={17} />
          </button>
        </div>

        {/* primary ------------------------------------------------ */}
        <div className="[grid-area:primary] flex items-center px-1">
          <textarea
            ref={textareaRef}
            name="prompt-textarea"
            aria-label={`Chat with ${modelLabel}`}
            placeholder="Ask anything"
            autoFocus
            rows={1}
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            onKeyDown={handlePromptKeyDown}
            onPaste={handlePaste}
            className="composer-textarea"
          />
        </div>

        {/* trailing ----------------------------------------------- */}
        <div className="[grid-area:trailing] flex items-end gap-1 pb-0.5">
          <button
            type="button"
            aria-label={listening ? "Stop dictation" : "Start dictation"}
            aria-pressed={listening}
            title={
              dictationSupported
                ? listening
                  ? "Stop dictation"
                  : "Dictate"
                : "Dictation isn't supported in this browser"
            }
            className="composer-btn"
            disabled={!dictationSupported || loading}
            style={listening ? { color: "var(--text-error, #ef4444)" } : undefined}
            onClick={toggleDictation}
          >
            <Icon name="mic" width={17} height={17} />
          </button>

          {loading ? (
            <button
              type="button"
              aria-label="Stop generating"
              data-testid="composer-stop-button"
              className="composer-send-btn"
              onClick={onStop}
            >
              <Icon name="stop" width={15} height={15} />
            </button>
          ) : (
            <button
              type="submit"
              aria-label="Send prompt"
              data-testid="composer-send-button"
              className="composer-send-btn"
              disabled={!canSubmit}
            >
              <Icon name="arrow-up" width={16} height={16} />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
