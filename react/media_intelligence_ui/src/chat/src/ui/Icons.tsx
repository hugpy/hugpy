/*
 * Icons.tsx — icon registry.
 *
 * Lucide-style single-color SVGs at viewBox 0 0 24 24. Width/height
 * default to 20 (match the `.icon` class); size from the caller wins.
 *
 * To add: append an entry to ICONS, then <Icon name="..." /> anywhere.
 */

import type { SVGProps } from "react";

type IconRenderer = (props: SVGProps<SVGSVGElement>) => JSX.Element;

const baseProps: SVGProps<SVGSVGElement> = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

function makeIcon(paths: JSX.Element): IconRenderer {
  return function Rendered(props: SVGProps<SVGSVGElement>) {
    return (
      <svg {...baseProps} {...props}>
        {paths}
      </svg>
    );
  };
}

const ICONS: Record<string, IconRenderer> = {
  "sidebar-toggle": makeIcon(
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <line x1="9" y1="4" x2="9" y2="20" />
    </>,
  ),

  menu: makeIcon(
    <>
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </>,
  ),

  "new-chat": makeIcon(
    <>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z" />
    </>,
  ),

  search: makeIcon(
    <>
      <circle cx="11" cy="11" r="7" />
      <line x1="20" y1="20" x2="16.65" y2="16.65" />
    </>,
  ),

  images: makeIcon(
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-5-5L5 21" />
    </>,
  ),

  user: makeIcon(
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0 1 16 0" />
    </>,
  ),

  plus: makeIcon(
    <>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </>,
  ),

  mic: makeIcon(
    <>
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </>,
  ),

  "arrow-up": makeIcon(
    <>
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </>,
  ),

  stop: makeIcon(<rect x="6" y="6" width="12" height="12" rx="2" />),

  copy: makeIcon(
    <>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </>,
  ),

  check: makeIcon(<polyline points="4 12 10 18 20 6" />),

  share: makeIcon(
    <>
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <line x1="8.6" y1="13.5" x2="15.4" y2="17.5" />
      <line x1="15.4" y1="6.5" x2="8.6" y2="10.5" />
    </>,
  ),

  edit: makeIcon(
    <>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5Z" />
    </>,
  ),

  more: makeIcon(
    <>
      <circle cx="5" cy="12" r="1.4" />
      <circle cx="12" cy="12" r="1.4" />
      <circle cx="19" cy="12" r="1.4" />
    </>,
  ),

  sources: makeIcon(
    <>
      <path d="M4 4h12a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H4Z" />
      <line x1="8" y1="9" x2="14" y2="9" />
      <line x1="8" y1="13" x2="14" y2="13" />
    </>,
  ),

  "switch-model": makeIcon(
    <>
      <polyline points="17 1 21 5 17 9" />
      <path d="M3 11V9a4 4 0 0 1 4-4h14" />
      <polyline points="7 23 3 19 7 15" />
      <path d="M21 13v2a4 4 0 0 1-4 4H3" />
    </>,
  ),

  "chevron-down": makeIcon(<polyline points="6 9 12 15 18 9" />),

  sparkles: makeIcon(
    <>
      <path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9Z" />
      <path d="M19 14l.9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9Z" />
    </>,
  ),

  // tiny dot used as a sidebar header "logo" placeholder
  "logo-dot": makeIcon(<circle cx="12" cy="12" r="5" fill="currentColor" />),
};

interface IconProps extends SVGProps<SVGSVGElement> {
  name: keyof typeof ICONS | string;
}

export function Icon({ name, ...rest }: IconProps): JSX.Element | null {
  const Renderer = ICONS[name];
  if (!Renderer) {
    if (typeof console !== "undefined") {
      console.warn("[Icon] unknown icon name", name);
    }
    return null;
  }
  return <Renderer {...rest} />;
}

export type IconName = keyof typeof ICONS;
