"""
tools/filesystem_tool.py
─────────────────────────
Safe filesystem operations for AIDA.
Only operates within user-approved paths; never touches system dirs.

Supported commands (natural language + Russian):
  create folder/directory  → mkdir
  delete folder/file       → rmdir / remove (only empty dirs, non-system files)
  list files               → listdir
  read file                → read text file
  write / create file      → write text file
  rename                   → rename
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool

log = logging.getLogger("aida.tools.filesystem")

# Directories we will never touch regardless of what the user says
_BLOCKED_ROOTS = {
    "c:\\windows", "c:\\program files", "c:\\program files (x86)",
    "/usr", "/bin", "/sbin", "/etc", "/lib", "/lib64", "/boot",
    "/dev", "/proc", "/sys", "/root",
}

_MAX_READ_BYTES = 50_000   # safety limit for file reads


def _is_safe_path(path: str) -> bool:
    try:
        resolved = str(Path(path).resolve()).lower()
        return not any(resolved.startswith(b) for b in _BLOCKED_ROOTS)
    except Exception:
        return False


# ── Russian disk-letter mapping ──────────────────────────────────────────────
_RU_DISK = {
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ж":"j","з":"z",
    "и":"i","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r",
    "с":"s","т":"t","у":"u","ф":"f","х":"h","ц":"c","э":"e","ю":"u",
}

_RE_QUOTED      = re.compile(r"""['"]([^'"]+)['"]""")
_RE_BARE_PATH   = re.compile(r"([A-Za-z]):\\([\w\\. \-]+)")
_RE_DISK_LETTER = re.compile(r"\b(?:disk|drive|диск[еа]?)\s+([A-Za-zА-Яа-я])", re.IGNORECASE)
_RE_ON_DRIVE    = re.compile(r"\bon\s+([A-Za-z])(?::|(?=\s|$))", re.IGNORECASE)
_RE_FOLDER_NAME = re.compile(
    r"\b(?:named|called|name(?:\s+it)?|с\s+именем|назови(?:\s+его)?|имя)\s+"
    r"[\"']?([A-Za-zА-Яа-я0-9][^\s,\"']*(?:\s+[A-Za-zА-Яа-я0-9][^\s,\"']*){0,4}?)[\"']?"
    r"(?:\s+(?:on|in|at|на|в)\b|[,.]|$)",
    re.IGNORECASE,
)
_RE_FOLDER_NOUN = re.compile(
    r"\b(?:folder|directory|папку?|директорию)\s+[\"']?([A-Za-zА-Яа-я0-9][^\s,\"']{0,40})[\"']?"
    r"(?:\s+(?:on|in|at|called|named|на|в)\b|$)",
    re.IGNORECASE,
)
_RE_IN_PATH     = re.compile(r"\bin\s+[\"']?([A-Za-z]:\\[\w\\ .\-]+)[\"']?", re.IGNORECASE)


def _parse_natural_path(text: str) -> Optional[str]:
    """
    Extract a Windows path from natural-language descriptions.
    Examples:
      "create folder Test on disk D"           → D:\\Test
      "make a folder D:\\Projects\\New"         → D:\\Projects\\New
      "создай папку Проект на диске Д"          → D:\\Проект
      "create folder named Reports in C:\\Work" → C:\\Work\\Reports
      "the folder on D called backup"           → D:\\backup
    """
    t = text.strip()

    # 1. Quoted path wins
    m = _RE_QUOTED.search(t)
    if m:
        return m.group(1).strip()

    # 2. Bare Windows path — only return immediately if no "name/called/named" keyword
    m = _RE_BARE_PATH.search(t)
    if m and not re.search(r"\b(?:named|called|name\s+it|с\s+именем|назови)\b", t, re.IGNORECASE):
        return m.group(0)
    # If there IS a name keyword, keep the bare path as parent and continue

    # 3. Disk letter
    disk_letter: Optional[str] = None
    dm = _RE_DISK_LETTER.search(t)
    if not dm:
        dm = _RE_ON_DRIVE.search(t)
    if not dm:  # bare "D:" anywhere
        dm = re.search(r"\b([A-Za-z]):[/\\]", t, re.IGNORECASE)
    if dm:
        letter = dm.group(1).lower()
        disk_letter = (_RU_DISK.get(letter, letter)).upper()

    # 4. Folder name
    nm = _RE_FOLDER_NAME.search(t)
    if not nm:
        nm = _RE_FOLDER_NOUN.search(t)
    folder_name = nm.group(1).strip() if nm else None

    # 5. Parent path
    pm = _RE_IN_PATH.search(t)
    parent_path = pm.group(1).strip() if pm else None

    # Build path
    if parent_path and folder_name:
        sep = "\\" if "\\" in parent_path else "/"
        return parent_path.rstrip("/\\") + sep + folder_name
    if parent_path:
        return parent_path
    if disk_letter and folder_name:
        return disk_letter + ":\\" + folder_name
    if disk_letter:
        return disk_letter + ":\\"
    if folder_name:
        return folder_name

    return None


def _extract_path(text: str) -> Optional[str]:
    """Alias kept for backward compat."""
    return _parse_natural_path(text)



class FilesystemTool(BaseTool):
    name        = "filesystem"
    description = ("Create, delete, list, read, write files and folders. "
                   "Operates only in safe (non-system) directories.")

    # ── keyword maps ──────────────────────────────────────────────────────────
    _CREATE_DIR  = ["create folder", "make folder", "new folder",
                    "create directory", "mkdir",
                    "создай папку", "создать папку", "новая папка"]
    _DELETE      = ["delete folder", "remove folder", "delete file", "remove file",
                    "удали папку", "удалить папку", "удали файл", "удалить файл"]
    _LIST        = ["list files", "show files", "what's in", "contents of",
                    "покажи файлы", "список файлов", "что в папке"]
    _READ        = ["read file", "open file", "show file", "contents of file",
                    "прочитай файл", "открой файл", "покажи содержимое"]
    _WRITE       = ["write file", "create file", "save file",
                    "создай файл", "запиши в файл", "сохрани файл"]
    _RENAME      = ["rename", "переименуй"]

    def _match(self, text: str, keywords) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in keywords)

    async def run(self, query: str, **kwargs) -> str:
        text = query.strip()
        if self._match(text, self._CREATE_DIR):
            return self._create_dir(text)
        if self._match(text, self._DELETE):
            return self._delete(text)
        if self._match(text, self._LIST):
            return self._list(text)
        if self._match(text, self._READ):
            return self._read(text)
        if self._match(text, self._WRITE):
            return self._write(text)
        if self._match(text, self._RENAME):
            return self._rename(text)
        return "❓ Filesystem: could not determine operation. Specify what to do and the path."

    # ── operations ────────────────────────────────────────────────────────────

    def _create_dir(self, text: str) -> str:
        path = _parse_natural_path(text)
        if not path:
            return "❌ Please specify the folder path to create."
        if not _is_safe_path(path):
            return f"🚫 Blocked: '{path}' is in a protected system directory."
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return f"✅ Folder created: {path}"
        except Exception as e:
            return f"❌ Could not create folder: {e}"

    def _delete(self, text: str) -> str:
        path = _parse_natural_path(text)
        if not path:
            return "❌ Please specify the path to delete."
        if not _is_safe_path(path):
            return f"🚫 Blocked: '{path}' is in a protected system directory."
        p = Path(path)
        try:
            if p.is_file():
                p.unlink()
                return f"✅ File deleted: {path}"
            elif p.is_dir():
                if any(p.iterdir()):
                    return (f"⚠️ Folder '{path}' is not empty. "
                            "Say 'delete all files in X' to confirm recursive delete.")
                p.rmdir()
                return f"✅ Folder deleted: {path}"
            else:
                return f"❌ Path not found: {path}"
        except Exception as e:
            return f"❌ Delete failed: {e}"

    def _list(self, text: str) -> str:
        path = _extract_path(text) or "."
        if not _is_safe_path(path):
            return f"🚫 Blocked: '{path}' is protected."
        try:
            entries = sorted(Path(path).iterdir(), key=lambda p: (p.is_file(), p.name))
            if not entries:
                return f"📂 '{path}' is empty."
            lines = []
            for e in entries[:50]:  # limit to 50 entries in response
                icon = "📄" if e.is_file() else "📁"
                lines.append(f"  {icon} {e.name}")
            header = f"📂 Contents of '{path}' ({len(entries)} items):\n"
            return header + "\n".join(lines) + ("" if len(entries) <= 50 else "\n  … (truncated)")
        except Exception as e:
            return f"❌ Cannot list '{path}': {e}"

    def _read(self, text: str) -> str:
        path = _extract_path(text)
        if not path:
            return "❌ Please specify the file to read."
        if not _is_safe_path(path):
            return f"🚫 Blocked: '{path}' is protected."
        try:
            data = Path(path).read_bytes()
            if len(data) > _MAX_READ_BYTES:
                data = data[:_MAX_READ_BYTES]
                truncated = True
            else:
                truncated = False
            content = data.decode("utf-8", errors="replace")
            note    = "\n… (file truncated at 50 KB)" if truncated else ""
            return f"📄 {path}:\n```\n{content}\n```{note}"
        except Exception as e:
            return f"❌ Cannot read '{path}': {e}"

    def _write(self, text: str) -> str:
        path = _extract_path(text)
        if not path:
            return "❌ Please specify the file path to write."
        if not _is_safe_path(path):
            return f"🚫 Blocked: '{path}' is protected."
        # Extract content between triple-quotes or after "with content"
        content_match = re.search(r'with content[:\s]+(.+)', text, re.DOTALL | re.IGNORECASE)
        content = content_match.group(1).strip() if content_match else ""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            return f"✅ File written: {path} ({len(content)} chars)"
        except Exception as e:
            return f"❌ Cannot write '{path}': {e}"

    def _rename(self, text: str) -> str:
        # Expect "rename X to Y"
        m = re.search(r'rename\s+(.+?)\s+to\s+(.+)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'переименуй\s+(.+?)\s+в\s+(.+)', text, re.IGNORECASE)
        if not m:
            return "❌ Usage: rename 'old_path' to 'new_path'"
        src, dst = m.group(1).strip().strip("'\""), m.group(2).strip().strip("'\"")
        if not _is_safe_path(src) or not _is_safe_path(dst):
            return "🚫 One or both paths are in a protected directory."
        try:
            Path(src).rename(dst)
            return f"✅ Renamed: '{src}' → '{dst}'"
        except Exception as e:
            return f"❌ Rename failed: {e}"
