/*
 * Disclaimer.tsx — small muted strip under the composer.
 */

interface DisclaimerProps {
  text?: string;
}

const DEFAULT_TEXT =
  "© 2026 hugpy · source-available — Your models, Your data.";

export default function Disclaimer({
  text = DEFAULT_TEXT,
}: DisclaimerProps): JSX.Element {
  return (
    <div className="w-full text-center text-[11px] text-token-text-tertiary py-2">
      {text}
    </div>
  );
}
