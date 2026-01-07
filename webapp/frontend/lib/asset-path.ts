/**
 * Builds absolute paths to assets that live under the Next.js public directory
 * while being aware of GitHub Pages base paths.
 */
type NextData = {
  assetPrefix?: string;
  nextExport?: { basePath?: string };
  nextExportOptions?: { basePath?: string };
};

const envBasePath = (process.env.NEXT_PUBLIC_BASE_PATH || '').replace(/\/$/, '');

const trimLeadingSlash = (value: string) => value.replace(/^\/+/, '');
const trimTrailingSlash = (value: string) => value.replace(/\/+/, '');

const joinBaseAndRelative = (base: string, relative: string) => {
  const safeBase = trimLeadingSlash(trimTrailingSlash(base));
  const safeRelative = trimLeadingSlash(relative);
  if (!safeBase) {
    return `/${safeRelative}`;
  }
  return `/${[safeBase, safeRelative].filter(Boolean).join('/')}`.replace(/\/{2,}/g, '/');
};

const resolveFromNextData = (relative: string): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  const nextData = (window as Window & { __NEXT_DATA__?: NextData }).__NEXT_DATA__;
  if (!nextData) {
    return null;
  }
  const base =
    nextData.nextExportOptions?.basePath ||
    nextData.nextExport?.basePath ||
    nextData.assetPrefix;
  if (!base) {
    return null;
  }
  return joinBaseAndRelative(base, relative);
};

const resolveFromLocation = (relative: string): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const baseUrl = new URL('.', window.location.href);
    const resolved = new URL(relative, baseUrl);
    return resolved.pathname.replace(/\/{2,}/g, '/');
  } catch {
    return null;
  }
};

export function makeAssetPath(relative: string): string {
  const normalizedRelative = trimLeadingSlash(relative);
  return (
    resolveFromNextData(normalizedRelative) ??
    resolveFromLocation(normalizedRelative) ??
    (envBasePath
      ? joinBaseAndRelative(envBasePath, normalizedRelative)
      : `/${normalizedRelative}`)
  );
}
