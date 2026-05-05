/**
 * Browser-side helpers for folder-upload to Brain.
 *
 * Walks a `<input webkitdirectory>` FileList or a drag-drop DataTransferItemList
 * (via webkitGetAsEntry recursion), filters out junk, and returns an upload
 * manifest. Filtering happens here so we never spend bandwidth on node_modules.
 *
 * Mirror of the backend's accept-list and ignore-list — keep them in sync with
 * gateway/app/routes/folder_ingest.py and gateway/app/config.py.
 */

const DEFAULT_IGNORE_NAMES: ReadonlySet<string> = new Set([
  "node_modules",
  ".git",
  ".next",
  "__pycache__",
  "dist",
  "build",
  "target",
  ".venv",
  "venv",
  ".idea",
  ".vscode",
  ".cache",
  ".DS_Store",
  "Thumbs.db",
]);

const DEFAULT_IGNORE_FILES: ReadonlySet<string> = new Set([
  "package-lock.json",
  "yarn.lock",
  "poetry.lock",
  "pnpm-lock.yaml",
]);

const DEFAULT_IGNORE_GLOBS: readonly RegExp[] = [
  /\.lock$/i,
];

const ALLOWED_EXTENSIONS: ReadonlySet<string> = new Set([
  "txt", "md", "markdown",
  "pdf", "docx",
  "py", "js", "ts", "tsx", "jsx",
  "json", "yaml", "yml", "toml",
  "html", "htm", "css", "sql", "sh",
  "go", "rs", "java", "c", "cpp", "h", "hpp",
  "rb", "php", "csv",
  "ini", "cfg", "conf",
]);

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB — must match document_max_file_size

export interface ManifestFile {
  file: File;
  relativePath: string;
  contentType: string | null;
  size: number;
}

export interface ManifestRejection {
  relativePath: string;
  reason: "ignored" | "too_large" | "unsupported_type";
}

export interface ManifestResult {
  accepted: ManifestFile[];
  rejected: ManifestRejection[];
}

export function shouldIgnorePath(relPath: string): boolean {
  const segments = relPath.split("/");
  for (const seg of segments) {
    if (DEFAULT_IGNORE_NAMES.has(seg)) return true;
  }
  const fname = segments[segments.length - 1] ?? relPath;
  if (DEFAULT_IGNORE_FILES.has(fname)) return true;
  return DEFAULT_IGNORE_GLOBS.some((re) => re.test(fname));
}

function classifyFile(
  file: File,
  relPath: string,
): "accept" | ManifestRejection["reason"] {
  if (shouldIgnorePath(relPath)) return "ignored";
  if (file.size > MAX_FILE_SIZE) return "too_large";
  const dot = relPath.lastIndexOf(".");
  const ext = dot >= 0 ? relPath.slice(dot + 1).toLowerCase() : "";
  if (!ext || !ALLOWED_EXTENSIONS.has(ext)) return "unsupported_type";
  return "accept";
}

function getRelativePath(file: File): string {
  const wkr = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return wkr && wkr.length > 0 ? wkr : file.name;
}

/** Build a manifest from `<input type="file" webkitdirectory>` selection. */
export function buildManifestFromInput(files: FileList): ManifestResult {
  const accepted: ManifestFile[] = [];
  const rejected: ManifestRejection[] = [];
  for (const f of Array.from(files)) {
    const rel = getRelativePath(f);
    const verdict = classifyFile(f, rel);
    if (verdict === "accept") {
      accepted.push({
        file: f,
        relativePath: rel,
        contentType: f.type || null,
        size: f.size,
      });
    } else {
      rejected.push({ relativePath: rel, reason: verdict });
    }
  }
  return { accepted, rejected };
}

/** Derive the human-friendly root folder name from a manifest. */
export function inferRootLabel(manifest: ManifestResult): string {
  if (manifest.accepted.length === 0) return "folder";
  const first = manifest.accepted[0].relativePath;
  const slash = first.indexOf("/");
  return slash > 0 ? first.slice(0, slash) : "folder";
}

// ── Drag-and-drop traversal ────────────────────────────────────────────

interface FileSystemEntryLike {
  isFile?: boolean;
  isDirectory?: boolean;
  name: string;
  file?: (cb: (f: File) => void, err?: (e: unknown) => void) => void;
  createReader?: () => {
    readEntries: (
      cb: (entries: FileSystemEntryLike[]) => void,
      err?: (e: unknown) => void,
    ) => void;
  };
}

function readAllEntries(
  reader: { readEntries: (cb: (entries: FileSystemEntryLike[]) => void, err?: (e: unknown) => void) => void },
): Promise<FileSystemEntryLike[]> {
  return new Promise((resolve, reject) => {
    const out: FileSystemEntryLike[] = [];
    const next = () => {
      reader.readEntries(
        (batch) => {
          if (batch.length === 0) {
            resolve(out);
            return;
          }
          out.push(...batch);
          // readEntries returns at most ~100 entries per call on Chromium/WebKit;
          // we MUST keep calling until it returns []. Don't optimize this away.
          next();
        },
        (err) => reject(err),
      );
    };
    next();
  });
}

async function walkEntry(
  entry: FileSystemEntryLike,
  parentPath: string,
  accepted: ManifestFile[],
  rejected: ManifestRejection[],
): Promise<void> {
  if (entry.isFile && entry.file) {
    const file: File = await new Promise((res, rej) =>
      entry.file!((f) => res(f), (e) => rej(e)),
    );
    const rel = parentPath ? `${parentPath}/${entry.name}` : entry.name;
    const verdict = classifyFile(file, rel);
    if (verdict === "accept") {
      accepted.push({
        file,
        relativePath: rel,
        contentType: file.type || null,
        size: file.size,
      });
    } else {
      rejected.push({ relativePath: rel, reason: verdict });
    }
    return;
  }
  if (entry.isDirectory && entry.createReader) {
    if (DEFAULT_IGNORE_NAMES.has(entry.name)) return; // prune early
    const reader = entry.createReader();
    const children = await readAllEntries(reader);
    const childPath = parentPath ? `${parentPath}/${entry.name}` : entry.name;
    for (const c of children) {
      await walkEntry(c, childPath, accepted, rejected);
    }
  }
}

export async function traverseDataTransferItems(
  items: DataTransferItemList,
): Promise<ManifestResult> {
  const accepted: ManifestFile[] = [];
  const rejected: ManifestRejection[] = [];
  const entries: FileSystemEntryLike[] = [];
  for (const item of Array.from(items)) {
    if (item.kind !== "file") continue;
    const get = (item as DataTransferItem & {
      webkitGetAsEntry?: () => FileSystemEntryLike | null;
    }).webkitGetAsEntry;
    const entry = get?.call(item) ?? null;
    if (entry) entries.push(entry);
  }
  for (const e of entries) {
    await walkEntry(e, "", accepted, rejected);
  }
  return { accepted, rejected };
}

/** Returns true if the drop event includes at least one folder entry. */
export function dropContainsFolder(items: DataTransferItemList): boolean {
  for (const item of Array.from(items)) {
    if (item.kind !== "file") continue;
    const get = (item as DataTransferItem & {
      webkitGetAsEntry?: () => FileSystemEntryLike | null;
    }).webkitGetAsEntry;
    const entry = get?.call(item) ?? null;
    if (entry?.isDirectory) return true;
  }
  return false;
}
